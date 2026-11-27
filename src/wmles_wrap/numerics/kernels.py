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
BC_INFLOW_TOTAL = 8  # subsonic inflow: total pressure, total temperature and flow direction

DISS_NONE, DISS_RUSANOV, DISS_ROE = 0, 1, 2
SGS_NONE, SGS_VREMAN, SGS_WALE, SGS_SMAGORINSKY = 0, 1, 2, 3
WM_NONE, WM_EQUILIBRIUM, WM_ALGEBRAIC = 0, 1, 2
WM_MAX_POINTS = 64

# statistics sample vector (float64): means and second moments of primitive variables
STAT_FIELDS = ("rho", "u", "v", "w", "p", "T", "uu", "vv", "ww", "uv", "uw", "vw", "pp", "TT", "rhou", "mut")
vec16d = wp.types.vector(len(STAT_FIELDS), wp.float64)
# wall-modeled face samples (float64): wall pressure (first cell), modeled traction, heat flux
WALL_FIELDS = ("p", "tw_x", "tw_y", "tw_z", "q")
vec5d = wp.types.vector(len(WALL_FIELDS), wp.float64)
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
    quarter = scalar(0.25)
    two = scalar(2.0)
    two_thirds = scalar(2.0 / 3.0)
    four_thirds = scalar(4.0 / 3.0)
    tenth = scalar(0.1)
    third = scalar(1.0 / 3.0)
    c_r04 = scalar(0.4)  # Reichardt constants
    c_r78 = scalar(7.8)
    c_r11 = scalar(1.0 / 11.0)
    c_r33 = scalar(1.0 / 33.0)
    vecN = wp.types.vector(WM_MAX_POINTS, scalar)
    # NOTE: fractional constants must be defined here (closure, exact in the kernel precision).
    # Inside a kernel, Python float literals are float32: scalar(2.0 / 3.0) there computes
    # 2/3 in float32 and then converts, silently losing precision in float64 kernels.
    tiny = scalar(1e-30)
    smooth_bc = scalar(0.05)  # width (in u_n / a) of the smooth far-field inflow/outflow blend

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
        sgs_model: int  # 0 none, 1 Vreman, 2 WALE, 3 Smagorinsky
        sgs_c: scalar
        wm_model: int  # 1 equilibrium ODE, 2 algebraic (Reichardt + Crocco-Busemann)
        wm_kappa: scalar
        wm_aplus: scalar
        wm_n: int
        wm_r: scalar
        wm_iters: int
        wm_iters_warm: int
        flux_type: int  # 0: KEEP-PE + sensor-weighted dissipation (LES), 1: Roe-MUSCL (steady RANS)
        muscl_e: vec5  # van Albada regularization (eps * reference)^2 per primitive variable
        limiter: int  # roe_muscl: 0 van Albada (kappa = 0 average, limited), 1 none (unlimited kappa-scheme)
        muscl_kappa: scalar  # unlimited kappa-scheme: 1/3 third-order upwind-biased (CFL3D), 0 Fromm, -1 upwind
        fix_linear: scalar  # Harten entropy fix on the entropy/shear waves (fraction of a), Roe-MUSCL
        steady: int  # 1: steady solver (solver.mode rans): smooth far-field inflow/outflow switch

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
    def euler_flux(W: vec5, S: vec3, p: Params) -> vec5:
        u = velocity(W)
        un = wp.dot(u, S)
        rho_un = W[0] * un
        m = rho_un * u + W[4] * S
        H = p.gamma / (p.gamma - one) * W[4] / W[0] + half * wp.dot(u, u)
        return vec5(rho_un, m[0], m[1], m[2], rho_un * H)

    @wp.func
    def van_albada(a: scalar, b: scalar, e: scalar) -> scalar:
        """Smooth limited slope (van Albada); unlimited average for |a|, |b| << sqrt(e)."""
        return (a * (b * b + e) + b * (a * a + e)) / (a * a + b * b + two * e)

    @wp.func
    def muscl(Wm: vec5, W0: vec5, Wp: vec5, sgn: scalar, p: Params) -> vec5:
        """Face state of cell W0 towards its neighbour Wp (sgn = +1) or Wm (sgn = -1)."""
        out = vec5()
        if p.limiter == 1:
            # unlimited kappa-scheme: W0 + sgn/4 [(1 - kappa) (away from the face) + (1 + kappa) (towards it)]
            for k in range(5):
                fwd = Wp[k] - W0[k]
                back = W0[k] - Wm[k]
                if sgn < zero:
                    fwd = W0[k] - Wm[k]
                    back = Wp[k] - W0[k]
                out[k] = W0[k] + sgn * quarter * ((one - p.muscl_kappa) * back + (one + p.muscl_kappa) * fwd)
        else:
            for k in range(5):
                out[k] = W0[k] + sgn * half * van_albada(W0[k] - Wm[k], Wp[k] - W0[k], p.muscl_e[k])
        return out

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
        if p.flux_type == 1 and p.fix_linear > zero:
            l2 = harten(unt, p.fix_linear * at)
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
            ut_ref = u_ref - un_ref * nrm
            if p.steady == 1:
                # steady solver: blend inflow/outflow smoothly in u_n (the switch at u_n = 0, e.g. on a
                # far field parallel to the flow, makes the residual non-smooth and stalls the
                # implicit iteration)
                wgt = half * (one + wp.tanh(un_b / (smooth_bc * a_b)))
                s = wgt * (W1[4] / wp.pow(W1[0], p.gamma)) + (one - wgt) * (p.w_inf[4] / wp.pow(p.w_inf[0], p.gamma))
                ut_ref = wgt * (u_i - un_i * nrm) + (one - wgt) * (u_o - un_o * nrm)
            rho_b = wp.pow(a_b * a_b / (p.gamma * s), one / g1)
            p_b = rho_b * a_b * a_b / p.gamma
            u_b = ut_ref + un_b * nrm
            Wb = vec5(rho_b, u_b[0], u_b[1], u_b[2], p_b)
        elif bctype == 5:  # subsonic inflow: rho, u from free stream, p extrapolated
            Wb = vec5(p.w_inf[0], p.w_inf[1], p.w_inf[2], p.w_inf[3], W1[4])
            a_o = wp.sqrt(p.gamma * p.w_inf[4] / p.w_inf[0])
            if wp.abs(wp.dot(velocity(p.w_inf), nrm)) > a_o:
                Wb = p.w_inf
        elif bctype == 8:
            # subsonic inflow, total conditions (Carlson, NASA/TM-2011-217181): p0 = p_exit,
            # T0 = t_wall, flow direction d = u_wall; the outgoing Riemann invariant
            # R+ = u_n + 2a/(gamma-1) comes from the interior, and a_b solves
            # a_b^2 + (gamma-1)/2 q^2 = a0^2 with u_n,b = q (d.n) = R+ - 2 a_b/(gamma-1)
            g1 = p.gamma - one
            cth = wp.min(wp.dot(u_wall, nrm), -tenth)  # d.n (n outward), kept away from tangential
            a_i = wp.sqrt(p.gamma * W1[4] / W1[0])
            rp = wp.dot(velocity(W1), nrm) + two * a_i / g1
            a02 = p.gamma * p.R * t_wall
            qa = cth * cth + two / g1
            qc = half * g1 * rp * rp - a02 * cth * cth
            a_b = (rp + wp.sqrt(wp.max(rp * rp - qa * qc, zero))) / qa
            q = wp.max((rp - two * a_b / g1) / cth, zero)
            q = wp.min(q, wp.sqrt(a02 / (half * g1 + one)))  # at most sonic
            T_b = t_wall - half * q * q / p.cp
            p_b = p_exit * wp.pow(T_b / t_wall, p.gamma / g1)
            Wb = vec5(p_b / (p.R * T_b), q * u_wall[0], q * u_wall[1], q * u_wall[2], p_b)
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
            # blocks thinner than ng cells (e.g. one-cell-thick quasi-2D grids): mirror the
            # last interior cell instead of reading the opposite ghost layers
            i = wp.min(wp.max(i, ng), ng + nc - 1)
            gi = cell(axis, g, t1, t2)
            ii = cell(axis, i, t1, t2)
            Wi = W[ii[0], ii[1], ii[2]]
            Wg = Wb
            if bctype == 0 or bctype == 1:
                ug = two * u_wall - velocity(Wi)
                Tg = temperature(Wi, p)
                if bctype == 1:
                    Tg = wp.max(two * t_wall - Tg, tenth * t_wall)
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
    def k_sgs(
        W: wp.array3d(dtype=vec5),
        GU: wp.array3d(dtype=mat33),
        vol: wp.array3d(dtype=scalar),
        p: Params,
        mut: wp.array3d(dtype=scalar),
    ):
        """SGS eddy viscosity mu_t = rho nu_t, filter width Delta = V^(1/3).
        Vreman (2004), WALE (Nicoud & Ducros 1999), static Smagorinsky."""
        a, b, c = wp.tid()
        i = a + 1
        j = b + 1
        k = c + 1
        g = GU[i, j, k]  # g[m, n] = d u_m / d x_n
        d2 = wp.pow(vol[i, j, k], two_thirds)
        nut = zero
        if p.sgs_model == 1:
            bb = d2 * (g * wp.transpose(g))  # beta_ij = Delta^2 sum_m alpha_mi alpha_mj, alpha_mi = g[i, m]
            B = (bb[0, 0] * bb[1, 1] - bb[0, 1] * bb[0, 1] + bb[0, 0] * bb[2, 2] - bb[0, 2] * bb[0, 2]
                 + bb[1, 1] * bb[2, 2] - bb[1, 2] * bb[1, 2])
            aa = wp.ddot(g, g)
            if aa > tiny and B > zero:
                nut = p.sgs_c * wp.sqrt(B / aa)
        elif p.sgs_model == 2:
            S = half * (g + wp.transpose(g))
            g2 = g * g
            tr = (g2[0, 0] + g2[1, 1] + g2[2, 2]) / scalar(3.0)
            Sd = half * (g2 + wp.transpose(g2)) - tr * wp.identity(n=3, dtype=scalar)
            ss = wp.ddot(S, S)
            sd = wp.ddot(Sd, Sd)
            den = wp.pow(ss, scalar(2.5)) + wp.pow(sd, scalar(1.25))
            if den > tiny:
                nut = p.sgs_c * p.sgs_c * d2 * wp.pow(sd, scalar(1.5)) / den
        elif p.sgs_model == 3:
            S = half * (g + wp.transpose(g))
            nut = p.sgs_c * p.sgs_c * d2 * wp.sqrt(two * wp.ddot(S, S))
        mut[i, j, k] = W[i, j, k][0] * nut

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
        wm_lo: int,
        wm_hi: int,
        tw_lo: wp.array2d(dtype=vec3),
        q_lo: wp.array2d(dtype=scalar),
        tw_hi: wp.array2d(dtype=vec3),
        q_hi: wp.array2d(dtype=scalar),
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
        wall_face = (fa == 0 and nodiss_lo == 1) or (fa == nfaces - 1 and nodiss_hi == 1)
        flux = vec5()
        if p.flux_type == 1:
            # Roe-MUSCL: van Albada primitive reconstruction from two cells per side; on wall/slip
            # faces first-order mirrored states and no dissipation (pressure flux only)
            if wall_face:
                flux = half * (euler_flux(WL, Sf, p) + euler_flux(WR, Sf, p))
            else:
                WLL = W[2 * iL - iR, 2 * jL - jR, 2 * kL - kR]
                WRR = W[2 * iR - iL, 2 * jR - jL, 2 * kR - kL]
                QL = muscl(WLL, WL, WR, one, p)
                QR = muscl(WL, WR, WRR, -one, p)
                if QL[0] <= zero or QL[4] <= zero or QR[0] <= zero or QR[4] <= zero:
                    QL = WL
                    QR = WR
                flux = half * (euler_flux(QL, Sf, p) + euler_flux(QR, Sf, p)) - half * roe_dissipation(QL, QR, Sf, p)
        else:
            flux = keep_pe_flux(WL, WR, Sf, p)

        if p.diss_type > 0 and p.flux_type == 0:
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
            if wall_face:
                mt = zero  # no eddy viscosity on wall/slip faces (ghost values are mirrors)
            mue = mu + mt
            kappa = p.cp * (mu / p.prandtl + mt / p.prandtl_t)
            div = gu[0, 0] + gu[1, 1] + gu[2, 2]
            tau = mue * (gu + wp.transpose(gu)) - (two_thirds * mue * div) * wp.identity(n=3, dtype=scalar)
            ts = tau * Sf
            fv = vec5(zero, ts[0], ts[1], ts[2], wp.dot(ts, uf) + kappa * wp.dot(gt, Sf))
            # wall-modeled faces: modeled traction tau_w * t and heat flux Q = k dT/dn (n into the
            # fluid) replace the resolved viscous flux; the sign accounts for S pointing into the
            # fluid on the low side and out of it on the high side
            t1 = b
            t2 = c
            if axis == 1:
                t1 = a
            elif axis == 2:
                t1 = a
                t2 = b
            area = wp.length(Sf)
            if fa == 0 and wm_lo == 1:
                tw = tw_lo[t1, t2] * area
                fv = vec5(zero, tw[0], tw[1], tw[2], q_lo[t1, t2] * area)
            if fa == nfaces - 1 and wm_hi == 1:
                tw = tw_hi[t1, t2] * area
                fv = -vec5(zero, tw[0], tw[1], tw[2], q_hi[t1, t2] * area)
            flux = flux - fv
        F[a, b, c] = flux

    # ---- wall models ------------------------------------------------------------------------
    @wp.func
    def reichardt(yp: scalar) -> scalar:
        return wp.log(one + c_r04 * yp) / c_r04 + c_r78 * (one - wp.exp(-yp * c_r11) - yp * c_r11 * wp.exp(-yp * third))

    @wp.func
    def reichardt_deriv(yp: scalar) -> scalar:
        return one / (one + c_r04 * yp) + c_r78 * (wp.exp(-yp * c_r11) * c_r11 - wp.exp(-yp * third) * c_r11
                                                    + yp * c_r33 * wp.exp(-yp * third))

    @wp.func
    def algebraic_utau(U: scalar, h: scalar, nu: scalar) -> scalar:
        ut = wp.max(wp.sqrt(nu * U / h), tiny)
        for it in range(30):
            yp = h * ut / nu
            g = ut * reichardt(yp) - U
            dg = reichardt(yp) + yp * reichardt_deriv(yp)
            ut = wp.max(ut - g / dg, tenth * ut)
        return ut

    @wp.func
    def wm_algebraic(U: scalar, h: scalar, T_e: scalar, pe: scalar, iso: int, T_w: scalar, p: Params) -> wp.vec2d:
        Tw = T_e
        if iso == 1:
            Tw = T_w
        rho_w = pe / (p.R * Tw)
        mu_w = viscosity(Tw, p)
        ut = algebraic_utau(U, h, mu_w / rho_w)
        tau = rho_w * ut * ut
        Q = scalar(0.0)
        if iso == 1 and U > zero:
            T_aw = T_e + wp.pow(p.prandtl, third) * U * U / (two * p.cp)
            Q = p.cp / p.prandtl * (T_aw - T_w) * tau / U
        return wp.vec2d(wp.float64(tau), wp.float64(Q))

    @wp.func
    def wm_equilibrium(U: scalar, h: scalar, T_e: scalar, pe: scalar, iso: int, T_w: scalar, tau0: scalar,
                       p: Params) -> wp.vec2d:
        n = p.wm_n
        r = p.wm_r
        dy0 = h / scalar(n - 1)
        if wp.abs(r - one) > scalar(1e-6):
            dy0 = h * (r - one) / (wp.pow(r, scalar(n - 1)) - one)
        tau = tau0
        iters = p.wm_iters_warm
        if tau0 <= zero:  # cold start from the algebraic model
            res = wm_algebraic(U, h, T_e, pe, iso, T_w, p)
            tau = scalar(res[0])
            iters = p.wm_iters
        T = vecN()
        Ig = vecN()
        Igu = vecN()
        # grid: y_0 = 0, y_i = y_{i-1} + seg_i, seg_i = dy0 r^(i-1), y_{n-1} = h
        y = scalar(0.0)
        seg = dy0
        for i in range(n):
            if i > 0:
                y = y + seg
                seg = seg * r
            T[i] = T_e
            if iso == 1:
                T[i] = T_w + (T_e - T_w) * y / h
        Q = scalar(0.0)
        for it in range(iters):
            # pass 1: momentum quadrature int dy / (mu + mu_t) -> tau_new
            y = scalar(0.0)
            seg = dy0
            cum = scalar(0.0)
            f_prev = scalar(0.0)
            for i in range(n):
                ds = scalar(0.0)
                if i > 0:
                    ds = seg
                    y = y + seg
                    seg = seg * r
                rho = pe / (p.R * T[i])
                mu = viscosity(T[i], p)
                dmp = one - wp.exp(-(y * wp.sqrt(rho * tau) / mu) / p.wm_aplus)
                f = one / (mu + p.wm_kappa * rho * y * wp.sqrt(tau / rho) * dmp * dmp)
                cum = cum + half * (f + f_prev) * ds
                f_prev = f
            tau_new = U / cum
            # pass 2: velocity profile and energy quadratures
            y = scalar(0.0)
            seg = dy0
            cu = scalar(0.0)
            f_prev = scalar(0.0)
            g_prev = scalar(0.0)
            gu_prev = scalar(0.0)
            for i in range(n):
                ds = scalar(0.0)
                if i > 0:
                    ds = seg
                    y = y + seg
                    seg = seg * r
                rho = pe / (p.R * T[i])
                mu = viscosity(T[i], p)
                dmp = one - wp.exp(-(y * wp.sqrt(rho * tau) / mu) / p.wm_aplus)
                mt = p.wm_kappa * rho * y * wp.sqrt(tau / rho) * dmp * dmp
                f = one / (mu + mt)
                cu = cu + half * (f + f_prev) * ds
                u = U * cu / cum
                g = one / (p.cp * (mu / p.prandtl + mt / p.prandtl_t))
                gu = tau_new * u * g
                if i == 0:
                    Ig[i] = zero
                    Igu[i] = zero
                else:
                    Ig[i] = Ig[i - 1] + half * (g + g_prev) * ds
                    Igu[i] = Igu[i - 1] + half * (gu + gu_prev) * ds
                f_prev = f
                g_prev = g
                gu_prev = gu
            if iso == 1:
                Q = (T_e - T_w + Igu[n - 1]) / Ig[n - 1]
                for i in range(n):
                    T[i] = T_w + Q * Ig[i] - Igu[i]
            else:
                Q = scalar(0.0)
                for i in range(n):
                    T[i] = T_e + (Igu[n - 1] - Igu[i])
            tau = tau_new  # undamped fixed point: contraction ~0.5 per iteration
        return wp.vec2d(wp.float64(tau), wp.float64(Q))

    @wp.kernel(module="unique")
    def k_wall_model(
        W: wp.array3d(dtype=vec5),
        xc: wp.array3d(dtype=vec3),
        axis: int,
        side: int,
        ng: int,
        nc: int,
        cell_index: int,
        h_dist: scalar,
        iso: int,
        T_w: scalar,
        warm: int,
        p: Params,
        tw_out: wp.array2d(dtype=vec3),
        q_out: wp.array2d(dtype=scalar),
        ut_out: wp.array2d(dtype=scalar),
    ):
        t1, t2 = wp.tid()
        first = ng
        ghost = ng - 1
        step = 1
        if side == 1:
            first = ng + nc - 1
            ghost = ng + nc
            step = -1
        cf = cell(axis, first, t1 + ng, t2 + ng)
        cg = cell(axis, ghost, t1 + ng, t2 + ng)
        xfirst = xc[cf[0], cf[1], cf[2]]
        xghost = xc[cg[0], cg[1], cg[2]]
        xf = half * (xfirst + xghost)
        nrm = wp.normalize(xfirst - xghost)  # into the fluid
        m = int(cell_index - 1)
        if h_dist > zero:
            m = int(0)
            for q in range(nc):
                cq = cell(axis, first + step * q, t1 + ng, t2 + ng)
                if wp.dot(xc[cq[0], cq[1], cq[2]] - xf, nrm) < h_dist:
                    m = q + 1
            m = wp.min(m, nc - 1)
        ce = cell(axis, first + step * m, t1 + ng, t2 + ng)
        We = W[ce[0], ce[1], ce[2]]
        h = wp.dot(xc[ce[0], ce[1], ce[2]] - xf, nrm)
        u = velocity(We)
        ut = u - wp.dot(u, nrm) * nrm
        U = wp.length(ut)
        tdir = vec3()
        if U > tiny:
            tdir = ut / U
        T_e = temperature(We, p)
        res = wp.vec2d()
        tau_prev = zero
        if warm == 1:
            tau_prev = wp.length(tw_out[t1, t2])
        if p.wm_model == 1:
            res = wm_equilibrium(U, h, T_e, We[4], iso, T_w, tau_prev, p)
        else:
            res = wm_algebraic(U, h, T_e, We[4], iso, T_w, p)
        tau = scalar(res[0])
        tw_out[t1, t2] = tau * tdir
        q_out[t1, t2] = scalar(res[1])
        Tw = T_e
        if iso == 1:
            Tw = T_w
        ut_out[t1, t2] = wp.sqrt(tau * p.R * Tw / We[4])

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
            nu = mu / Wc[0] * wp.max(four_thirds, p.gamma / p.prandtl)
            lam = lam + two * nu * (wp.dot(sx, sx) + wp.dot(sy, sy) + wp.dot(sz, sz)) / V
        wp.atomic_min(dtmin, 0, cfl * V / lam)

    @wp.kernel(module="unique")
    def k_momentum(U: wp.array3d(dtype=vec5), d: vec3, vol: wp.array3d(dtype=scalar), ng: int,
                   out: wp.array3d(dtype=wp.float64)):
        a, b, c = wp.tid()
        Uc = U[a + ng, b + ng, c + ng]
        out[a, b, c] = wp.float64((Uc[1] * d[0] + Uc[2] * d[1] + Uc[3] * d[2]) * vol[a + ng, b + ng, c + ng])

    @wp.kernel(module="unique")
    def k_add_momentum(U: wp.array3d(dtype=vec5), d: vec3, dm: scalar, ng: int):
        """Add a uniform momentum increment dm * d per unit volume, keeping internal energy."""
        a, b, c = wp.tid()
        Uc = U[a + ng, b + ng, c + ng]
        m0 = vec3(Uc[1], Uc[2], Uc[3])
        m1 = m0 + dm * d
        Uc[1] = m1[0]
        Uc[2] = m1[1]
        Uc[3] = m1[2]
        Uc[4] = Uc[4] + half * (wp.dot(m1, m1) - wp.dot(m0, m0)) / Uc[0]
        U[a + ng, b + ng, c + ng] = Uc

    @wp.kernel(module="unique")
    def k_stats(W: wp.array3d(dtype=vec5), mut: wp.array3d(dtype=scalar), ng: int, p: Params,
                acc: wp.array3d(dtype=vec16d)):
        a, b, c = wp.tid()
        Wc = W[a + ng, b + ng, c + ng]
        r = wp.float64(Wc[0])
        u = wp.float64(Wc[1])
        v = wp.float64(Wc[2])
        w = wp.float64(Wc[3])
        pr = wp.float64(Wc[4])
        T = wp.float64(temperature(Wc, p))
        smp = vec16d(r, u, v, w, pr, T, u * u, v * v, w * w, u * v, u * w, v * w, pr * pr, T * T, r * u,
                     wp.float64(mut[a + ng, b + ng, c + ng]))
        acc[a, b, c] = acc[a, b, c] + smp

    @wp.kernel(module="unique")
    def k_wall_sample(
        W: wp.array3d(dtype=vec5),
        S: wp.array3d(dtype=vec3),
        axis: int,
        side: int,
        ng: int,
        nc: int,
        tw: wp.array2d(dtype=vec3),
        q: wp.array2d(dtype=scalar),
        p_inf: scalar,
        accumulate: int,
        force: wp.array2d(dtype=vec3),
        acc: wp.array2d(dtype=vec5d),
    ):
        """Per wall-modeled face: force exerted by the fluid on the wall (modeled shear plus
        pressure relative to p_inf, wall pressure = first-cell pressure since the slip-mirror
        ghost has zero normal pressure gradient) and, if ``accumulate``, float64 sums of
        WALL_FIELDS."""
        t1, t2 = wp.tid()
        fp = ng
        first = ng
        sgn = scalar(1.0)
        if side == 1:
            fp = ng + nc
            first = ng + nc - 1
            sgn = scalar(-1.0)
        fi = cell(axis, fp, t1 + ng, t2 + ng)
        Sn = sgn * S[fi[0], fi[1], fi[2]]  # area vector pointing into the fluid (out of the body)
        ci = cell(axis, first, t1 + ng, t2 + ng)
        pw = W[ci[0], ci[1], ci[2]][4]
        t = tw[t1, t2]
        force[t1, t2] = t * wp.length(Sn) - (pw - p_inf) * Sn
        if accumulate == 1:
            acc[t1, t2] = acc[t1, t2] + vec5d(wp.float64(pw), wp.float64(t[0]), wp.float64(t[1]),
                                              wp.float64(t[2]), wp.float64(q[t1, t2]))

    return {
        "Params": Params,
        "mat33": mat33,
        "prim": k_prim,
        "set_cons": k_set_cons,
        "bc": k_bc,
        "gradients": k_gradients,
        "sgs": k_sgs,
        "flux": k_flux,
        "residual": k_residual,
        "rk": k_rk,
        "dt": k_dt,
        "momentum": k_momentum,
        "add_momentum": k_add_momentum,
        "stats": k_stats,
        "wall_model": k_wall_model,
        "wall_sample": k_wall_sample,
    }
