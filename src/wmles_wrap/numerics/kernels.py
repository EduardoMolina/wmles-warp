"""Warp kernels for the compressible Navier-Stokes finite-volume discretization.

Arrays (per sub-block, padded with ng ghost layers unless noted):
    U   vec5  conservative [rho, rho u, rho v, rho w, rho E]
    W   vec5  primitive    [rho, u, v, w, p]
    GU  mat33 velocity gradient, GU[i, j] = d u_i / d x_j     (valid on padded range minus 1)
    GT  vec3  temperature gradient                            (valid on padded range minus 1)
    th  scalar Ducros sensor theta                            (valid on padded range minus 1)
    mut scalar SGS eddy viscosity
    Fi/Fj/Fk  vec5 face fluxes on interior faces, logical shape (nc + e_axis)
    R   vec5  residual dU/dt * V on interior cells, logical shape nc

Inviscid flux: KEEP-PE split form (Shima, Kuya, Tamaki & Kawai, JCP 2021), which preserves
kinetic energy and pressure equilibrium. Shock capturing: Roe (Harten entropy fix) or
Rusanov dissipation weighted by phi = max(theta_L, theta_R), where theta is the Ducros
sensor, by default multiplied by a Jameson-type pressure second-difference sensor and
restricted to compression (Ducros alone is O(1) in irrotational smooth flow and degrades the
scheme to first order there). Dissipation is switched off on wall/slip faces. Viscous flux: face gradients from averaged Green-Gauss
cell gradients with a directional (cell-to-cell) correction.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

import warp as wp

from ..precision import Precision, kernel_factory

# boundary condition codes (bc kernel)
BC_WALL_ADIABATIC = 0
BC_WALL_ISOTHERMAL = 1
BC_SLIP = 2
BC_FARFIELD = 3
BC_EXTRAPOLATE = 4
BC_INFLOW = 5
BC_OUTFLOW = 6
BC_DIRICHLET = 7  # filled on the host (verification)

DISS_NONE, DISS_RUSANOV, DISS_ROE = 0, 1, 2
VISC_SUTHERLAND, VISC_POWER, VISC_CONSTANT = 0, 1, 2


@kernel_factory
def make_kernels(prec: Precision):
    scalar = prec.scalar
    vec3 = prec.vec3
    vec5 = prec.vec5
    mat33 = wp.types.matrix((3, 3), scalar)
    zero = scalar(0.0)
    one = scalar(1.0)
    half = scalar(0.5)
    two = scalar(2.0)
    two_thirds = scalar(2.0 / 3.0)
    tiny = scalar(1e-30)

    @wp.struct
    class Params:
        gamma: scalar
        R: scalar
        cv: scalar
        cp: scalar
        prandtl: scalar
        prandtl_t: scalar
        visc_model: int
        s_nd: scalar
        power_exp: scalar
        inv_re: scalar
        w_inf: vec5
        sensor_eps: scalar
        sensor_threshold: scalar
        entropy_fix: scalar
        diss_type: int
        use_sensor: int  # 0: none (phi = 1), 1: Ducros, 2: Ducros x pressure sensor (compression only)
        pressure_gain: scalar
        viscous: int

    # ---- thermodynamics -----------------------------------------------------------------
    @wp.func
    def cons_to_prim(U: vec5, p: Params) -> vec5:
        rho = U[0]
        u = U[1] / rho
        v = U[2] / rho
        w = U[3] / rho
        pr = (p.gamma - one) * (U[4] - half * rho * (u * u + v * v + w * w))
        return vec5(rho, u, v, w, pr)

    @wp.func
    def prim_to_cons(W: vec5, p: Params) -> vec5:
        rho = W[0]
        ke = half * rho * (W[1] * W[1] + W[2] * W[2] + W[3] * W[3])
        return vec5(rho, rho * W[1], rho * W[2], rho * W[3], W[4] / (p.gamma - one) + ke)

    @wp.func
    def temperature(W: vec5, p: Params) -> scalar:
        return W[4] / (W[0] * p.R)

    @wp.func
    def velocity(W: vec5) -> vec3:
        return vec3(W[1], W[2], W[3])

    @wp.func
    def viscosity(T: scalar, p: Params) -> scalar:
        m = one
        if p.visc_model == 0:
            m = wp.pow(T, scalar(1.5)) * (one + p.s_nd) / (T + p.s_nd)
        elif p.visc_model == 1:
            m = wp.pow(T, p.power_exp)
        return m * p.inv_re

    @wp.func
    def cell(axis: int, n: int, t1: int, t2: int) -> wp.vec3i:
        """3D index from a normal index along ``axis`` and two tangential indices."""
        if axis == 0:
            return wp.vec3i(n, t1, t2)
        if axis == 1:
            return wp.vec3i(t1, n, t2)
        return wp.vec3i(t1, t2, n)

    # ---- inviscid fluxes ----------------------------------------------------------------
    @wp.func
    def keep_pe_flux(WL: vec5, WR: vec5, S: vec3, p: Params) -> vec5:
        uL = velocity(WL)
        uR = velocity(WR)
        rho_m = half * (WL[0] + WR[0])
        u_m = half * (uL + uR)
        p_m = half * (WL[4] + WR[4])
        un = wp.dot(u_m, S)
        C = rho_m * un
        fm = C * u_m + p_m * S
        fe = (
            C * half * wp.dot(uL, uR)
            + p_m * un / (p.gamma - one)
            + half * (WL[4] * wp.dot(uR, S) + WR[4] * wp.dot(uL, S))
        )
        return vec5(C, fm[0], fm[1], fm[2], fe)

    @wp.func
    def rusanov_dissipation(WL: vec5, WR: vec5, S: vec3, p: Params) -> vec5:
        area = wp.length(S)
        n = S / area
        aL = wp.sqrt(p.gamma * WL[4] / WL[0])
        aR = wp.sqrt(p.gamma * WR[4] / WR[0])
        lam = wp.max(wp.abs(wp.dot(velocity(WL), n)) + aL, wp.abs(wp.dot(velocity(WR), n)) + aR)
        return (lam * area) * (prim_to_cons(WR, p) - prim_to_cons(WL, p))

    @wp.func
    def harten(lam: scalar, delta: scalar) -> scalar:
        l = wp.abs(lam)
        if l < delta:
            l = (l * l + delta * delta) / (two * delta)
        return l

    @wp.func
    def roe_dissipation(WL: vec5, WR: vec5, S: vec3, p: Params) -> vec5:
        """|A_n| (U_R - U_L) |S| with Roe averages (Toro, ch. 11)."""
        area = wp.length(S)
        n = S / area
        g1 = p.gamma - one
        uL = velocity(WL)
        uR = velocity(WR)
        HL = p.gamma / g1 * WL[4] / WL[0] + half * wp.dot(uL, uL)
        HR = p.gamma / g1 * WR[4] / WR[0] + half * wp.dot(uR, uR)
        sL = wp.sqrt(WL[0])
        sR = wp.sqrt(WR[0])
        rt = sL * sR
        ut = (sL * uL + sR * uR) / (sL + sR)
        Ht = (sL * HL + sR * HR) / (sL + sR)
        q2 = wp.dot(ut, ut)
        at2 = wp.max(g1 * (Ht - half * q2), tiny)
        at = wp.sqrt(at2)
        unt = wp.dot(ut, n)
        drho = WR[0] - WL[0]
        dp = WR[4] - WL[4]
        du = uR - uL
        dun = wp.dot(du, n)
        delta = p.entropy_fix * at
        l1 = harten(unt - at, delta)
        l2 = wp.abs(unt)
        l5 = harten(unt + at, delta)
        a1 = (dp - rt * at * dun) / (two * at2)
        a5 = (dp + rt * at * dun) / (two * at2)
        a2 = drho - dp / at2
        dut = du - dun * n
        d0 = l1 * a1 + l2 * a2 + l5 * a5
        dm = l1 * a1 * (ut - at * n) + l2 * (a2 * ut + rt * dut) + l5 * a5 * (ut + at * n)
        de = (
            l1 * a1 * (Ht - at * unt)
            + l2 * (a2 * half * q2 + rt * (wp.dot(ut, du) - unt * dun))
            + l5 * a5 * (Ht + at * unt)
        )
        return area * vec5(d0, dm[0], dm[1], dm[2], de)

    # ---- kernels --------------------------------------------------------------------------
    @wp.kernel(module="unique")
    def k_prim(U: wp.array3d(dtype=vec5), ng: int, p: Params, W: wp.array3d(dtype=vec5)):
        a, b, c = wp.tid()
        W[a + ng, b + ng, c + ng] = cons_to_prim(U[a + ng, b + ng, c + ng], p)

    @wp.kernel(module="unique")
    def k_set_cons(W: wp.array3d(dtype=vec5), p: Params, U: wp.array3d(dtype=vec5)):
        a, b, c = wp.tid()
        U[a, b, c] = prim_to_cons(W[a, b, c], p)

    @wp.kernel(module="unique")
    def k_bc(
        W: wp.array3d(dtype=vec5),
        S: wp.array3d(dtype=vec3),
        axis: int,
        side: int,
        ng: int,
        nc: int,
        bctype: int,
        t_wall: scalar,
        u_wall: vec3,
        p_exit: scalar,
        p: Params,
    ):
        t1, t2 = wp.tid()
        fp = ng
        first = ng
        if side == 1:
            fp = ng + nc
            first = ng + nc - 1
        fi = cell(axis, fp, t1, t2)
        Sf = S[fi[0], fi[1], fi[2]]
        nrm = Sf / wp.max(wp.length(Sf), tiny)
        if side == 0:
            nrm = -nrm  # outward
        ci = cell(axis, first, t1, t2)
        W1 = W[ci[0], ci[1], ci[2]]

        # boundary state for characteristic-type conditions (same for all ghost layers)
        Wb = W1
        if bctype == 3:  # far field, Riemann invariants
            g1 = p.gamma - one
            u_i = velocity(W1)
            u_o = velocity(p.w_inf)
            a_i = wp.sqrt(p.gamma * W1[4] / W1[0])
            a_o = wp.sqrt(p.gamma * p.w_inf[4] / p.w_inf[0])
            un_i = wp.dot(u_i, nrm)
            un_o = wp.dot(u_o, nrm)
            r_out = un_o + two * a_o / g1
            if un_i + a_i > zero:
                r_out = un_i + two * a_i / g1
            r_in = un_i - two * a_i / g1
            if un_i - a_i < zero:
                r_in = un_o - two * a_o / g1
            un_b = half * (r_out + r_in)
            a_b = scalar(0.25) * g1 * (r_out - r_in)
            ref = W1
            u_ref = u_i
            un_ref = un_i
            if un_b < zero:  # inflow: entropy and tangential velocity from free stream
                ref = p.w_inf
                u_ref = u_o
                un_ref = un_o
            s = ref[4] / wp.pow(ref[0], p.gamma)
            rho_b = wp.pow(a_b * a_b / (p.gamma * s), one / g1)
            p_b = rho_b * a_b * a_b / p.gamma
            u_b = u_ref + (un_b - un_ref) * nrm
            Wb = vec5(rho_b, u_b[0], u_b[1], u_b[2], p_b)
        elif bctype == 5:  # subsonic inflow: rho, u from free stream, p extrapolated
            Wb = vec5(p.w_inf[0], p.w_inf[1], p.w_inf[2], p.w_inf[3], W1[4])
            a_o = wp.sqrt(p.gamma * p.w_inf[4] / p.w_inf[0])
            if wp.abs(wp.dot(velocity(p.w_inf), nrm)) > a_o:
                Wb = p.w_inf
        elif bctype == 6:  # outflow: p imposed unless supersonic
            a_i = wp.sqrt(p.gamma * W1[4] / W1[0])
            if wp.dot(velocity(W1), nrm) < a_i:
                Wb = vec5(W1[0], W1[1], W1[2], W1[3], p_exit)

        for m in range(1, ng + 1):
            g = ng - m
            i = ng + m - 1
            if side == 1:
                g = ng + nc - 1 + m
                i = ng + nc - m
            gi = cell(axis, g, t1, t2)
            ii = cell(axis, i, t1, t2)
            Wi = W[ii[0], ii[1], ii[2]]
            Wg = Wb
            if bctype == 0 or bctype == 1:
                ug = two * u_wall - velocity(Wi)
                Tg = temperature(Wi, p)
                if bctype == 1:
                    Tg = wp.max(two * t_wall - Tg, scalar(0.1) * t_wall)
                Wg = vec5(Wi[4] / (p.R * Tg), ug[0], ug[1], ug[2], Wi[4])
            elif bctype == 2:
                ui = velocity(Wi)
                ug = ui - two * wp.dot(ui, nrm) * nrm
                Wg = vec5(Wi[0], ug[0], ug[1], ug[2], Wi[4])
            if bctype != 7:
                W[gi[0], gi[1], gi[2]] = Wg

    @wp.kernel(module="unique")
    def k_gradients(
        W: wp.array3d(dtype=vec5),
        Si: wp.array3d(dtype=vec3),
        Sj: wp.array3d(dtype=vec3),
        Sk: wp.array3d(dtype=vec3),
        vol: wp.array3d(dtype=scalar),
        p: Params,
        GU: wp.array3d(dtype=mat33),
        GT: wp.array3d(dtype=vec3),
        th: wp.array3d(dtype=scalar),
    ):
        a, b, c = wp.tid()
        i = a + 1
        j = b + 1
        k = c + 1
        Wc = W[i, j, k]
        uc = velocity(Wc)
        Tc = temperature(Wc, p)
        gu = mat33()
        gt = vec3()
        psi = zero  # Jameson-type sensor: max_d |q+ - 2q + q-| / (q+ + 2q + q-), q = p and rho
        pm = zero
        rm = zero
        for d in range(3):
            for s in range(2):
                o = 2 * s - 1  # neighbour offset -1 / +1
                ni = i
                nj = j
                nk = k
                Sf = vec3()
                if d == 0:
                    ni = i + o
                    Sf = Si[i + s, j, k]
                elif d == 1:
                    nj = j + o
                    Sf = Sj[i, j + s, k]
                else:
                    nk = k + o
                    Sf = Sk[i, j, k + s]
                if s == 0:
                    Sf = -Sf
                Wn = W[ni, nj, nk]
                uf = half * (uc + velocity(Wn))
                Tf = half * (Tc + temperature(Wn, p))
                gu = gu + wp.outer(uf, Sf)
                gt = gt + Tf * Sf
                if s == 0:
                    pm = Wn[4]
                    rm = Wn[0]
                else:
                    pp = Wn[4]
                    rp = Wn[0]
                    psi = wp.max(psi, wp.abs(pp - two * Wc[4] + pm) / (pp + two * Wc[4] + pm))
                    psi = wp.max(psi, wp.abs(rp - two * Wc[0] + rm) / (rp + two * Wc[0] + rm))
        inv_v = one / vol[i, j, k]
        gu = gu * inv_v
        GU[i, j, k] = gu
        GT[i, j, k] = gt * inv_v
        div = gu[0, 0] + gu[1, 1] + gu[2, 2]
        wx = gu[2, 1] - gu[1, 2]
        wy = gu[0, 2] - gu[2, 0]
        wz = gu[1, 0] - gu[0, 1]
        d2 = div * div
        w2 = wx * wx + wy * wy + wz * wz
        theta = d2 / (d2 + w2 + p.sensor_eps)
        if p.use_sensor == 2:
            # Regularized Ducros (-> 1 where there are no velocity gradients, e.g. an initial
            # discontinuity at rest) times the pressure sensor, compression only. Shocks have
            # O(1) pressure second differences, smooth flow O(h^2), so the scheme stays second
            # order away from shocks while vortical regions remain dissipation-free.
            theta = (d2 + p.sensor_eps) / (d2 + w2 + p.sensor_eps)
            if div > wp.sqrt(p.sensor_eps):
                theta = zero
            theta = theta * wp.min(one, p.pressure_gain * psi)
        th[i, j, k] = theta

    @wp.kernel(module="unique")
    def k_flux(
        W: wp.array3d(dtype=vec5),
        GU: wp.array3d(dtype=mat33),
        GT: wp.array3d(dtype=vec3),
        th: wp.array3d(dtype=scalar),
        mut: wp.array3d(dtype=scalar),
        S: wp.array3d(dtype=vec3),
        xc: wp.array3d(dtype=vec3),
        axis: int,
        ng: int,
        nfaces: int,
        nodiss_lo: int,
        nodiss_hi: int,
        p: Params,
        F: wp.array3d(dtype=vec5),
    ):
        a, b, c = wp.tid()
        iR = a + ng
        jR = b + ng
        kR = c + ng
        iL = iR
        jL = jR
        kL = kR
        fa = a
        if axis == 0:
            iL = iR - 1
        elif axis == 1:
            jL = jR - 1
            fa = b
        else:
            kL = kR - 1
            fa = c
        WL = W[iL, jL, kL]
        WR = W[iR, jR, kR]
        Sf = S[iR, jR, kR]
        flux = keep_pe_flux(WL, WR, Sf, p)

        if p.diss_type > 0:
            phi = one
            if p.use_sensor > 0:
                phi = wp.max(th[iL, jL, kL], th[iR, jR, kR])
                if phi <= p.sensor_threshold:
                    phi = zero
            if fa == 0 and nodiss_lo == 1:
                phi = zero
            if fa == nfaces - 1 and nodiss_hi == 1:
                phi = zero
            if phi > zero:
                if p.diss_type == 2:
                    flux = flux - (half * phi) * roe_dissipation(WL, WR, Sf, p)
                else:
                    flux = flux - (half * phi) * rusanov_dissipation(WL, WR, Sf, p)

        if p.viscous == 1:
            uL = velocity(WL)
            uR = velocity(WR)
            TL = temperature(WL, p)
            TR = temperature(WR, p)
            d = xc[iR, jR, kR] - xc[iL, jL, kL]
            dd = wp.dot(d, d)
            gu = half * (GU[iL, jL, kL] + GU[iR, jR, kR])
            gt = half * (GT[iL, jL, kL] + GT[iR, jR, kR])
            gu = gu + wp.outer((uR - uL) - gu * d, d / dd)
            gt = gt + ((TR - TL) - wp.dot(gt, d)) / dd * d
            Tf = half * (TL + TR)
            uf = half * (uL + uR)
            mu = viscosity(Tf, p)
            mt = half * (mut[iL, jL, kL] + mut[iR, jR, kR])
            mue = mu + mt
            kappa = p.cp * (mu / p.prandtl + mt / p.prandtl_t)
            div = gu[0, 0] + gu[1, 1] + gu[2, 2]
            tau = mue * (gu + wp.transpose(gu)) - (two_thirds * mue * div) * wp.identity(n=3, dtype=scalar)
            ts = tau * Sf
            flux = flux - vec5(zero, ts[0], ts[1], ts[2], wp.dot(ts, uf) + kappa * wp.dot(gt, Sf))
        F[a, b, c] = flux

    @wp.kernel(module="unique")
    def k_residual(
        Fi: wp.array3d(dtype=vec5),
        Fj: wp.array3d(dtype=vec5),
        Fk: wp.array3d(dtype=vec5),
        W: wp.array3d(dtype=vec5),
        vol: wp.array3d(dtype=scalar),
        src: wp.array3d(dtype=vec5),
        has_src: int,
        force: vec3,
        ng: int,
        R: wp.array3d(dtype=vec5),
    ):
        a, b, c = wp.tid()
        res = -(Fi[a + 1, b, c] - Fi[a, b, c] + Fj[a, b + 1, c] - Fj[a, b, c] + Fk[a, b, c + 1] - Fk[a, b, c])
        V = vol[a + ng, b + ng, c + ng]
        if has_src == 1:
            res = res + V * src[a, b, c]
        Wc = W[a + ng, b + ng, c + ng]
        rv = Wc[0] * V
        res = res + vec5(zero, rv * force[0], rv * force[1], rv * force[2], rv * wp.dot(velocity(Wc), force))
        R[a, b, c] = res

    @wp.kernel(module="unique")
    def k_rk(
        U: wp.array3d(dtype=vec5),
        U0: wp.array3d(dtype=vec5),
        R: wp.array3d(dtype=vec5),
        vol: wp.array3d(dtype=scalar),
        ca: scalar,
        cb: scalar,
        dt: scalar,
        ng: int,
    ):
        a, b, c = wp.tid()
        i = a + ng
        j = b + ng
        k = c + ng
        U[i, j, k] = ca * U0[i, j, k] + cb * (U[i, j, k] + (dt / vol[i, j, k]) * R[a, b, c])

    @wp.kernel(module="unique")
    def k_dt(
        W: wp.array3d(dtype=vec5),
        mut: wp.array3d(dtype=scalar),
        Si: wp.array3d(dtype=vec3),
        Sj: wp.array3d(dtype=vec3),
        Sk: wp.array3d(dtype=vec3),
        vol: wp.array3d(dtype=scalar),
        cfl: scalar,
        ng: int,
        p: Params,
        dtmin: wp.array(dtype=scalar),
    ):
        a, b, c = wp.tid()
        i = a + ng
        j = b + ng
        k = c + ng
        Wc = W[i, j, k]
        u = velocity(Wc)
        snd = wp.sqrt(p.gamma * Wc[4] / Wc[0])
        sx = half * (Si[i, j, k] + Si[i + 1, j, k])
        sy = half * (Sj[i, j, k] + Sj[i, j + 1, k])
        sz = half * (Sk[i, j, k] + Sk[i, j, k + 1])
        lam = (
            wp.abs(wp.dot(u, sx)) + snd * wp.length(sx)
            + wp.abs(wp.dot(u, sy)) + snd * wp.length(sy)
            + wp.abs(wp.dot(u, sz)) + snd * wp.length(sz)
        )
        V = vol[i, j, k]
        if p.viscous == 1:
            mu = viscosity(temperature(Wc, p), p) + mut[i, j, k]
            nu = mu / Wc[0] * wp.max(scalar(4.0 / 3.0), p.gamma / p.prandtl)
            lam = lam + two * nu * (wp.dot(sx, sx) + wp.dot(sy, sy) + wp.dot(sz, sz)) / V
        wp.atomic_min(dtmin, 0, cfl * V / lam)

    return {
        "Params": Params,
        "mat33": mat33,
        "prim": k_prim,
        "set_cons": k_set_cons,
        "bc": k_bc,
        "gradients": k_gradients,
        "flux": k_flux,
        "residual": k_residual,
        "rk": k_rk,
        "dt": k_dt,
    }
