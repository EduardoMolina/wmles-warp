"""Spalart-Allmaras turbulence model (negative SA, Allmaras, Johnson & Spalart, ICCFD7 2012),
segregated from the mean flow (RANS stage R2, docs/plan_rans.md).

The working variable nut (nu tilde, nondimensional like nu = mu*/rho) is a padded scalar per
sub-block in the residual precision. Model, as on the NASA TMR "SA-neg" page (no trip term,
with f_t2):

    D nut/Dt = P - D + 1/sigma [div((nu + nut f_n) grad nut) + c_b2 |grad nut|^2]
    nut >= 0:  P = c_b1 (1 - f_t2) S~ nut,   D = (c_w1 f_w - c_b1 f_t2 / kappa^2) (nut / d)^2
    nut <  0:  P = c_b1 (1 - c_t3) Omega nut, D = -c_w1 (nut / d)^2,  f_n = (c_n1 + chi^3)/(c_n1 - chi^3)
    S~ = Omega + S_bar if S_bar >= -c_v2 Omega, else Omega + Omega (c_v2^2 Omega + c_v3 S_bar) /
         ((c_v3 - 2 c_v2) Omega - S_bar),   S_bar = nut f_v2 / (kappa^2 d^2)
    mu_t = rho nut f_v1 for nut > 0, else 0.

Discretization (cell-centred, first order in convection as in the TMR verification codes):
    convection    sum_f min(u_f.S, 0) (nut_j - nut_i)                           (upwind)
    diffusion     1/sigma sum_f [nu_f + nut~_f + c_b2 (nut_f - nut_i)] (nut_j - nut_i) (S.d)/(d.d)
                  with nut~ = nut f_n, from div((nu + nut f_n + c_b2 nut) grad nut) - c_b2 nut lap nut
    source        V (P - D), Omega = |curl u| from the Green-Gauss gradients, d = wall distance.
Boundary ghosts: walls -nut_i (nut = 0 on the wall), slip +nut_i, far field/inflow nut_inf where
the flow enters and the first cell where it leaves (blended smoothly in u_n/a over 0.05),
outflow/extrapolation the first cell.

Implicit step (after the mean-flow update, same pseudo-time step and line structure): scalar
backward Euler, (V/dtau + dRes/dnut) dnut = R_nut, with upwind convection, frozen-coefficient
diffusion (coefficients clipped at 0), and the positive part of the full d(D - P)/dnut (finite
difference in the residual precision) on the diagonal.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

import numpy as np
import warp as wp

from ..precision import MixedPrecision, kernel_factory
from .kernels import (BC_DIRICHLET, BC_EXTRAPOLATE, BC_FARFIELD, BC_INFLOW, BC_OUTFLOW, BC_SLIP,
                      BC_WALL_ADIABATIC, BC_WALL_ISOTHERMAL, make_kernels)

# model constants
CB1, SIGMA, CB2, KAPPA = 0.1355, 2.0 / 3.0, 0.622, 0.41
CW2, CW3, CV1, CT3, CT4, CN1, CV2, CV3 = 0.3, 2.0, 7.1, 1.2, 0.5, 16.0, 0.7, 0.9
CW1 = CB1 / KAPPA**2 + (1.0 + CB2) / SIGMA


def sa_source_numpy(nt, om, nu, d):
    """P - D (numpy, float64), identical to the kernel's source_value (verification)."""
    nt, om, nu, d = (np.asarray(a, dtype=np.float64) for a in (nt, om, nu, d))
    kd2 = KAPPA**2 * d * d
    chi = nt / nu
    fv1 = chi**3 / (chi**3 + CV1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    sbar = nt * fv2 / kd2
    st = np.where(sbar >= -CV2 * om, om + sbar,
                  om + om * (CV2**2 * om + CV3 * sbar) / ((CV3 - 2 * CV2) * om - sbar))
    ft2 = CT3 * np.exp(-CT4 * chi**2)
    r = np.where(st > 1e-30, np.minimum(nt / np.where(st > 1e-30, st, 1.0) / kd2, 10.0), 10.0)
    g = r + CW2 * (r**6 - r)
    fw = g * ((1 + CW3**6) / (g**6 + CW3**6)) ** (1 / 6)
    pos = CB1 * (1 - ft2) * st * nt - (CW1 * fw - CB1 / KAPPA**2 * ft2) * (nt / d) ** 2
    neg = CB1 * (1 - CT3) * om * nt + CW1 * (nt / d) ** 2
    return np.where(nt >= 0, pos, neg)


@kernel_factory
def sa_kernels(mp: MixedPrecision):
    R = mp.residual
    rs, rv3, rv5 = R.scalar, R.vec3, R.vec5
    s = mp.solve.scalar
    K = make_kernels(R)
    Params, mat33 = K["Params"], K["mat33"]
    zero, one, half, two = rs(0.0), rs(1.0), rs(0.5), rs(2.0)
    c15, sixth, four_thirds = rs(1.5), rs(1.0 / 6.0), rs(4.0 / 3.0)
    cb1, sigma, cb2, kappa = rs(CB1), rs(SIGMA), rs(CB2), rs(KAPPA)
    cw1, cw2, cw3, cv1 = rs(CW1), rs(CW2), rs(CW3), rs(CV1)
    ct3, ct4, cn1, cv2, cv3 = rs(CT3), rs(CT4), rs(CN1), rs(CV2), rs(CV3)
    cw3_6 = rs(CW3**6)
    cv1_3 = rs(CV1**3)
    rmax = rs(10.0)
    smooth_bc = rs(0.05)
    fd_rel = rs(1e-6)
    tiny = rs(1e-30)
    wall_a, wall_i, slip = BC_WALL_ADIABATIC, BC_WALL_ISOTHERMAL, BC_SLIP
    farfield, inflow, dirichlet = BC_FARFIELD, BC_INFLOW, BC_DIRICHLET
    extrap, outflow = BC_EXTRAPOLATE, BC_OUTFLOW

    @wp.func
    def visc(T: rs, p: Params):
        m = one
        if p.visc_model == 0:
            m = wp.pow(T, c15) * (one + p.s_nd) / (T + p.s_nd)
        elif p.visc_model == 1:
            m = wp.pow(T, p.power_exp)
        return m * p.inv_re

    @wp.func
    def nu_lam(W: rv5, p: Params):
        return visc(W[4] / (W[0] * p.R), p) / W[0]

    @wp.func
    def fv1(chi: rs):
        c3 = chi * chi * chi
        return c3 / (c3 + cv1_3)

    @wp.func
    def nut_diff(nt: rs, nu: rs):
        """nut f_n (negative SA diffusion coefficient part)."""
        out = nt
        if nt < zero:
            c3 = (nt / nu) * (nt / nu) * (nt / nu)
            out = nt * (cn1 + c3) / (cn1 - c3)
        return out

    @wp.func
    def cell(axis: int, n: int, t1: int, t2: int):
        if axis == 0:
            return wp.vec3i(n, t1, t2)
        if axis == 1:
            return wp.vec3i(t1, n, t2)
        return wp.vec3i(t1, t2, n)

    @wp.func
    def face_S(Si: wp.array3d(dtype=rv3), Sj: wp.array3d(dtype=rv3), Sk: wp.array3d(dtype=rv3),
               idx: wp.vec3i, a: int, side: int):
        f = wp.vec3i(idx[0], idx[1], idx[2])
        if side == 1:
            f[a] = f[a] + 1
        S = rv3()
        if a == 0:
            S = Si[f[0], f[1], f[2]]
        elif a == 1:
            S = Sj[f[0], f[1], f[2]]
        else:
            S = Sk[f[0], f[1], f[2]]
        if side == 0:
            S = -S
        return S

    @wp.func
    def source_value(nt: rs, om: rs, nu: rs, d: rs):
        """P - D of the negative SA model."""
        kd2 = kappa * kappa * d * d
        src = rs(0.0)
        if nt >= zero:
            chi = nt / nu
            f1 = fv1(chi)
            fv2 = one - chi / (one + chi * f1)
            sbar = nt * fv2 / kd2
            st = om + sbar
            if sbar < -cv2 * om:
                st = om + om * (cv2 * cv2 * om + cv3 * sbar) / ((cv3 - two * cv2) * om - sbar)
            ft2 = ct3 * wp.exp(-ct4 * chi * chi)
            r = rmax
            if st > tiny:
                r = wp.min(nt / (st * kd2), rmax)
            g = r + cw2 * (wp.pow(r, rs(6.0)) - r)
            fw = g * wp.pow((one + cw3_6) / (wp.pow(g, rs(6.0)) + cw3_6), sixth)
            src = cb1 * (one - ft2) * st * nt - (cw1 * fw - cb1 / (kappa * kappa) * ft2) * (nt / d) * (nt / d)
        else:
            src = cb1 * (one - ct3) * om * nt + cw1 * (nt / d) * (nt / d)
        return src

    @wp.func
    def source(nt: rs, W: rv5, G: mat33, d: rs, p: Params):
        """(P - D, max(-d(P - D)/dnut, 0)); the derivative by a float64 one-sided difference, so
        f_w(r(nut)), S~(nut) and f_t2 are included (dropping them under-estimates the destruction
        derivative near walls by more than half and the iteration oscillates at high CFL)."""
        om = wp.length(rv3(G[2, 1] - G[1, 2], G[0, 2] - G[2, 0], G[1, 0] - G[0, 1]))
        nu = nu_lam(W, p)
        src = source_value(nt, om, nu, d)
        h = fd_rel * (wp.abs(nt) + nu)
        dsrc = (source_value(nt + h, om, nu, d) - src) / h
        return wp.vector(src, wp.max(-dsrc, zero))

    @wp.func
    def face_coefs(Wi: rv5, Wj: rv5, nti: rs, ntj: rs, xi: rv3, xj: rv3, S: rv3, p: Params):
        """(min(u_f.S, 0), diffusion coefficient [..] (S.d)/(d.d) / sigma) for face i -> j."""
        uf = half * rv3(Wi[1] + Wj[1], Wi[2] + Wj[2], Wi[3] + Wj[3])
        vn = wp.min(wp.dot(uf, S), zero)
        dd = xj - xi
        w = wp.dot(S, dd) / wp.dot(dd, dd)
        nui = nu_lam(Wi, p)
        nuj = nu_lam(Wj, p)
        coef = half * (nui + nuj) + half * (nut_diff(nti, nui) + nut_diff(ntj, nuj)) + cb2 * (half * (nti + ntj) - nti)
        return wp.vector(vn, coef * w / sigma)

    @wp.func
    def ghost_factor(code: int, Wi: rv5, S: rv3, p: Params):
        """B in dnut_ghost = B dnut_i for the implicit operator (0: explicit)."""
        b = zero
        if code == wall_a or code == wall_i:
            b = -one
        elif code == slip or code == extrap or code == outflow:
            b = one
        elif code == farfield or code == inflow:
            un = wp.dot(rv3(Wi[1], Wi[2], Wi[3]), S / wp.length(S))
            b = half * (one + wp.tanh(un / (smooth_bc * wp.sqrt(p.gamma * Wi[4] / Wi[0]))))
        return b

    # ---- ghosts, eddy viscosity, residual -----------------------------------------------------
    @wp.kernel(module="unique")
    def k_ghost(nut: wp.array3d(dtype=rs), W: wp.array3d(dtype=rv5), S: wp.array3d(dtype=rv3), axis: int, side: int,
                ng: int, nc: int, code: int, nut_inf: rs, p: Params):
        t1, t2 = wp.tid()
        fp = ng
        first = ng
        if side == 1:
            fp = ng + nc
            first = ng + nc - 1
        fi = cell(axis, fp, t1, t2)
        Sf = S[fi[0], fi[1], fi[2]]
        if side == 0:
            Sf = -Sf
        ci = cell(axis, first, t1, t2)
        W1 = W[ci[0], ci[1], ci[2]]
        n1 = nut[ci[0], ci[1], ci[2]]
        # far field/inflow: nut_inf where the flow enters, extrapolation where it leaves, blended
        # smoothly in u_n (as the mean-flow far field) so the residual stays smooth at u_n = 0
        un = wp.dot(rv3(W1[1], W1[2], W1[3]), Sf / wp.length(Sf))
        wgt = half * (one + wp.tanh(un / (smooth_bc * wp.sqrt(p.gamma * W1[4] / W1[0]))))
        for m in range(1, ng + 1):
            g = ng - m
            i = ng + m - 1
            if side == 1:
                g = ng + nc - 1 + m
                i = ng + nc - m
            i = wp.min(wp.max(i, ng), ng + nc - 1)
            gi = cell(axis, g, t1, t2)
            ii = cell(axis, i, t1, t2)
            v = n1
            if code == wall_a or code == wall_i:
                v = -nut[ii[0], ii[1], ii[2]]
            elif code == slip:
                v = nut[ii[0], ii[1], ii[2]]
            elif code == dirichlet:
                v = nut_inf
            elif code == farfield or code == inflow:
                v = wgt * n1 + (one - wgt) * nut_inf
            nut[gi[0], gi[1], gi[2]] = v

    @wp.kernel(module="unique")
    def k_mut(W: wp.array3d(dtype=rv5), nut: wp.array3d(dtype=rs), p: Params, mut: wp.array3d(dtype=rs)):
        i, j, k = wp.tid()
        Wc = W[i, j, k]
        nt = nut[i, j, k]
        m = zero
        if nt > zero:
            m = Wc[0] * nt * fv1(nt / nu_lam(Wc, p))
        mut[i, j, k] = m

    @wp.kernel(module="unique")
    def k_residual(
        W: wp.array3d(dtype=rv5),
        GU: wp.array3d(dtype=mat33),
        nut: wp.array3d(dtype=rs),
        dwall: wp.array3d(dtype=rs),
        xc: wp.array3d(dtype=rv3),
        Si: wp.array3d(dtype=rv3),
        Sj: wp.array3d(dtype=rv3),
        Sk: wp.array3d(dtype=rv3),
        vol: wp.array3d(dtype=rs),
        ng: int,
        p: Params,
        Rn: wp.array3d(dtype=rs),
    ):
        a0, b0, c0 = wp.tid()
        idx = wp.vec3i(a0 + ng, b0 + ng, c0 + ng)
        Wi = W[idx[0], idx[1], idx[2]]
        nti = nut[idx[0], idx[1], idx[2]]
        xi = xc[idx[0], idx[1], idx[2]]
        acc = rs(0.0)
        for a in range(3):
            for side in range(2):
                nb = wp.vec3i(idx[0], idx[1], idx[2])
                if side == 0:
                    nb[a] = nb[a] - 1
                else:
                    nb[a] = nb[a] + 1
                S = face_S(Si, Sj, Sk, idx, a, side)
                ntj = nut[nb[0], nb[1], nb[2]]
                c = face_coefs(Wi, W[nb[0], nb[1], nb[2]], nti, ntj, xi, xc[nb[0], nb[1], nb[2]], S, p)
                acc = acc + (c[1] - c[0]) * (ntj - nti)
        V = vol[idx[0], idx[1], idx[2]]
        src = source(nti, Wi, GU[idx[0], idx[1], idx[2]], dwall[idx[0], idx[1], idx[2]], p)
        Rn[a0, b0, c0] = acc + V * src[0]

    # ---- implicit step --------------------------------------------------------------------------
    @wp.kernel(module="unique")
    def k_assemble(
        W: wp.array3d(dtype=rv5),
        GU: wp.array3d(dtype=mat33),
        nut: wp.array3d(dtype=rs),
        mut: wp.array3d(dtype=rs),
        dwall: wp.array3d(dtype=rs),
        xc: wp.array3d(dtype=rv3),
        Si: wp.array3d(dtype=rv3),
        Sj: wp.array3d(dtype=rv3),
        Sk: wp.array3d(dtype=rv3),
        vol: wp.array3d(dtype=rs),
        cfl: rs,
        ng: int,
        nc: wp.vec3i,
        la: int,
        fcode: wp.array(dtype=int),
        p: Params,
        Dn: wp.array3d(dtype=s),
        Ln: wp.array3d(dtype=s),
        Un: wp.array3d(dtype=s),
    ):
        a0, b0, c0 = wp.tid()
        idx = wp.vec3i(a0 + ng, b0 + ng, c0 + ng)
        Wi = W[idx[0], idx[1], idx[2]]
        nti = nut[idx[0], idx[1], idx[2]]
        xi = xc[idx[0], idx[1], idx[2]]
        V = vol[idx[0], idx[1], idx[2]]
        ui = rv3(Wi[1], Wi[2], Wi[3])
        snd = wp.sqrt(p.gamma * Wi[4] / Wi[0])
        lam = rs(0.0)
        ssum = rs(0.0)
        for a in range(3):
            sa = half * (face_S(Si, Sj, Sk, idx, a, 1) - face_S(Si, Sj, Sk, idx, a, 0))
            lam = lam + wp.abs(wp.dot(ui, sa)) + snd * wp.length(sa)
            ssum = ssum + wp.dot(sa, sa)
        mu = visc(Wi[4] / (Wi[0] * p.R), p) + mut[idx[0], idx[1], idx[2]]
        lam = lam + two * mu / Wi[0] * wp.max(four_thirds, p.gamma / p.prandtl) * ssum / V
        diag = lam / cfl
        lo = rs(0.0)
        up = rs(0.0)
        for a in range(3):
            for side in range(2):
                nb = wp.vec3i(idx[0], idx[1], idx[2])
                if side == 0:
                    nb[a] = nb[a] - 1
                else:
                    nb[a] = nb[a] + 1
                S = face_S(Si, Sj, Sk, idx, a, side)
                c = face_coefs(Wi, W[nb[0], nb[1], nb[2]], nti, nut[nb[0], nb[1], nb[2]], xi,
                               xc[nb[0], nb[1], nb[2]], S, p)
                dc = wp.max(c[1], zero)
                own = dc - c[0]  # dRes_i/dnut_i
                off = c[0] - dc  # dRes_i/dnut_j
                diag = diag + own
                inside = nb[a] >= ng and nb[a] < ng + nc[a]
                code = fcode[2 * a + side]
                if (not inside) and code >= 0:
                    diag = diag + off * ghost_factor(code, Wi, S, p)
                elif a == la and inside:
                    if side == 0:
                        lo = off
                    else:
                        up = off
        src = source(nti, Wi, GU[idx[0], idx[1], idx[2]], dwall[idx[0], idx[1], idx[2]], p)
        diag = diag + V * src[1]
        Dn[idx[0], idx[1], idx[2]] = s(diag)
        Ln[idx[0], idx[1], idx[2]] = s(lo)
        Un[idx[0], idx[1], idx[2]] = s(up)

    @wp.kernel(module="unique")
    def k_line(
        W: wp.array3d(dtype=rv5),
        nut: wp.array3d(dtype=rs),
        xc: wp.array3d(dtype=rv3),
        Si: wp.array3d(dtype=rv3),
        Sj: wp.array3d(dtype=rv3),
        Sk: wp.array3d(dtype=rv3),
        ng: int,
        nc: wp.vec3i,
        la: int,
        fcode: wp.array(dtype=int),
        p: Params,
        Dn: wp.array3d(dtype=s),
        Ln: wp.array3d(dtype=s),
        Un: wp.array3d(dtype=s),
        Rn: wp.array3d(dtype=rs),
        dn_in: wp.array3d(dtype=s),
        color: int,
        offset: int,
        G: wp.array3d(dtype=s),
        Y: wp.array3d(dtype=s),
        dn_out: wp.array3d(dtype=s),
    ):
        t1, t2 = wp.tid()
        if color >= 0 and (offset + t1 + t2) % 2 != color:
            return
        n = nc[la]
        gp = s(0.0)
        yp = s(0.0)
        for c in range(n):
            idx = cell(la, c + ng, t1 + ng, t2 + ng)
            Wi = W[idx[0], idx[1], idx[2]]
            nti = nut[idx[0], idx[1], idx[2]]
            xi = xc[idx[0], idx[1], idx[2]]
            r = s(Rn[idx[0] - ng, idx[1] - ng, idx[2] - ng])
            for a in range(3):
                for side in range(2):
                    nb = wp.vec3i(idx[0], idx[1], idx[2])
                    if side == 0:
                        nb[a] = nb[a] - 1
                    else:
                        nb[a] = nb[a] + 1
                    inside = nb[a] >= ng and nb[a] < ng + nc[a]
                    code = fcode[2 * a + side]
                    use = (not (a == la and inside)) and (inside or code < 0)
                    if use:
                        S = face_S(Si, Sj, Sk, idx, a, side)
                        cf = face_coefs(Wi, W[nb[0], nb[1], nb[2]], nti, nut[nb[0], nb[1], nb[2]], xi,
                                        xc[nb[0], nb[1], nb[2]], S, p)
                        r = r - s(cf[0] - wp.max(cf[1], zero)) * dn_in[nb[0], nb[1], nb[2]]
            dc = Dn[idx[0], idx[1], idx[2]]
            l = Ln[idx[0], idx[1], idx[2]]
            if c > 0:
                dc = dc - l * gp
                r = r - l * yp
            gp = Un[idx[0], idx[1], idx[2]] / dc
            yp = r / dc
            G[idx[0], idx[1], idx[2]] = gp
            Y[idx[0], idx[1], idx[2]] = yp
        x = s(yp)
        last = cell(la, n - 1 + ng, t1 + ng, t2 + ng)
        dn_out[last[0], last[1], last[2]] = x
        for cc in range(n - 1):
            c = n - 2 - cc
            idx = cell(la, c + ng, t1 + ng, t2 + ng)
            x = Y[idx[0], idx[1], idx[2]] - G[idx[0], idx[1], idx[2]] * x
            dn_out[idx[0], idx[1], idx[2]] = x

    @wp.kernel(module="unique")
    def k_update(dn: wp.array3d(dtype=s), ng: int, nut: wp.array3d(dtype=rs)):
        a0, b0, c0 = wp.tid()
        nut[a0 + ng, b0 + ng, c0 + ng] = nut[a0 + ng, b0 + ng, c0 + ng] + rs(dn[a0 + ng, b0 + ng, c0 + ng])

    return k_ghost, k_mut, k_residual, k_assemble, k_line, k_update


class SAModel:
    """SA state, ghosts, eddy viscosity, residual and implicit step for one Solver."""

    def __init__(self, solver):
        from .implicit import line_layout

        self.s = solver
        self.mp = solver_mixed(solver)
        (self.k_ghost, self.k_mut, self.k_residual, self.k_assemble, self.k_line,
         self.k_update) = sa_kernels(self.mp)
        g = solver.gas
        self.nut_inf = solver.cfg.rans.nut_inf_ratio * float(g.mu(np.float64(1.0))) / 1.0  # rho_inf = 1
        ng = solver.ng
        rp, sp = self.mp.residual, self.mp.solve
        self.layout = line_layout(solver)
        self.exact_fn = None  # verification: x (N, 3) -> nut at dirichlet ghosts
        self.source_fn = None  # verification: x (N, 3) -> nut forcing per unit volume (steady)
        self._src = {}
        self.nut, self.R, self.D, self.L, self.U, self.G, self.Y, self.dn, self.dn2 = ({} for _ in range(9))
        for sb in solver.subs:
            pad = sb.padded_shape_cells(ng)
            dev = sb.device
            self.nut[sb.id] = wp.full(pad, rp.scalar(self.nut_inf), dtype=rp.scalar, device=dev)
            self.R[sb.id] = wp.zeros(sb.nc, dtype=rp.scalar, device=dev)
            for arr in (self.D, self.L, self.U, self.G, self.Y, self.dn, self.dn2):
                arr[sb.id] = wp.zeros(pad, dtype=sp.scalar, device=dev)

    # ---- called from Integrator.rhs ---------------------------------------------------------
    def fill_ghosts(self) -> None:
        s, ng = self.s, self.s.ng
        s.halo.exchange(self.nut, stages=(0,))
        rs = self.mp.residual.scalar
        for sid, d in s.integrator.data.items():
            m = s.metrics[sid]
            S = (m.Si, m.Sj, m.Sk)
            for b in d.bcs:
                wp.launch(self.k_ghost, dim=b.dims,
                          inputs=[self.nut[sid], d.W, S[b.axis], b.axis, b.side, ng, d.sub.nc[b.axis], b.code,
                                  rs(self.nut_inf), s.integrator.params], device=d.sub.device)
            if self.exact_fn is not None:
                from ..parallel.halo import _kernels as halo_kernels

                scatter = halo_kernels(self.mp.residual.scalar)[1]
                for dr in d.dirichlet:
                    vals = wp.array(np.asarray(self.exact_fn(dr.xc)).astype(self.mp.residual.np_dtype),
                                    dtype=rs, device=d.sub.device)
                    wp.launch(scatter, dim=len(dr.xc), inputs=[vals, dr.idx], outputs=[self.nut[sid]],
                              device=d.sub.device)
        s.halo.exchange(self.nut, stages=(1, 2))

    def eddy_viscosity(self, sid: int, d) -> None:
        wp.launch(self.k_mut, dim=d.W.shape, inputs=[d.W, self.nut[sid], self.s.integrator.params], outputs=[d.mut],
                  device=d.sub.device)

    # ---- residual and implicit step -----------------------------------------------------------
    def residual(self) -> None:
        """R_nut at the state of the last Integrator.rhs (ghosts, W, GU are current)."""
        s, ng = self.s, self.s.ng
        dw = s.wall_distance().d
        for sid, d in s.integrator.data.items():
            m = s.metrics[sid]
            wp.launch(self.k_residual, dim=d.sub.nc,
                      inputs=[d.W, d.GU, self.nut[sid], dw[sid], m.xc, m.Si, m.Sj, m.Sk, m.vol, ng,
                              s.integrator.params], outputs=[self.R[sid]], device=d.sub.device)
            if self.source_fn is not None:
                if sid not in self._src:
                    xc = d.xc_interior.reshape(-1, 3)
                    v = s.interior(m.vol.numpy()).astype(np.float64)
                    self._src[sid] = (v * np.asarray(self.source_fn(xc)).reshape(d.sub.nc))
                self.R[sid] = wp.array((self.R[sid].numpy() + self._src[sid]).astype(self.mp.residual.np_dtype),
                                       dtype=self.mp.residual.scalar, device=d.sub.device)

    def residual_norm(self) -> float:
        tot, n = 0.0, 0
        for sid in self.R:
            v = self.s.interior(self.s.metrics[sid].vol.numpy()).astype(np.float64)
            r = self.R[sid].numpy().astype(np.float64) / v
            tot += float((r**2).sum())
            n += r.size
        return float(np.sqrt(tot / max(n, 1))) / self.nut_inf

    def implicit_step(self, cfl: float, rcfg, zebra: bool) -> None:
        from .implicit import sweep_converged

        s, ng = self.s, self.s.ng
        dw = s.wall_distance().d
        P = s.integrator.params
        rs = self.mp.residual.scalar
        for sid, d in s.integrator.data.items():
            L, m = self.layout[sid], s.metrics[sid]
            wp.launch(self.k_assemble, dim=d.sub.nc,
                      inputs=[d.W, d.GU, self.nut[sid], d.mut, dw[sid], m.xc, m.Si, m.Sj, m.Sk, m.vol, rs(cfl), ng,
                              wp.vec3i(*d.sub.nc), L.la, L.fcode, P],
                      outputs=[self.D[sid], self.L[sid], self.U[sid]], device=d.sub.device)
            self.dn[sid].zero_()
            self.dn2[sid].zero_()
        for sweep in range(rcfg.sweeps):
            check = sweep + 1 >= rcfg.min_sweeps
            if check and zebra:
                for sid in self.dn:
                    wp.copy(self.dn2[sid], self.dn[sid])
            for color in ((0, 1) if zebra else (-1,)):
                for sid, d in s.integrator.data.items():
                    L, m = self.layout[sid], s.metrics[sid]
                    out = self.dn[sid] if zebra else self.dn2[sid]
                    wp.launch(self.k_line, dim=L.dims,
                              inputs=[d.W, self.nut[sid], m.xc, m.Si, m.Sj, m.Sk, ng, wp.vec3i(*d.sub.nc), L.la,
                                      L.fcode, P, self.D[sid], self.L[sid], self.U[sid], self.R[sid], self.dn[sid],
                                      color, L.offset, self.G[sid], self.Y[sid]],
                              outputs=[out], device=d.sub.device)
                if not zebra:
                    self.dn, self.dn2 = self.dn2, self.dn
                s.halo.exchange(self.dn, stages=(0,))
            if check and sweep_converged([(self.dn[i], self.dn2[i]) for i in self.dn], ng, rcfg.sweep_tolerance):
                break
        for sid, d in s.integrator.data.items():
            wp.launch(self.k_update, dim=d.sub.nc, inputs=[self.dn[sid], ng], outputs=[self.nut[sid]],
                      device=d.sub.device)

    # ---- parent-block layout (restart, output) --------------------------------------------------
    def parent_nut(self) -> list[np.ndarray]:
        s = self.s
        out = [np.zeros(tuple(n - 1 for n in x.shape[:3])) for x in s.parent_blocks]
        for sb in s.subs:
            lo, hi = sb.lo, sb.hi
            out[sb.parent][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = s.interior(self.nut[sb.id].numpy())
        return out

    def load_parent_nut(self, arrays: list[np.ndarray]) -> None:
        s, g = self.s, self.s.ng
        for sb in s.subs:
            a = np.full(sb.padded_shape_cells(g), self.nut_inf)
            lo, hi = sb.lo, sb.hi
            a[g:-g, g:-g, g:-g] = arrays[sb.parent][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            self.nut[sb.id] = wp.array(a.astype(self.mp.residual.np_dtype), dtype=self.mp.residual.scalar,
                                       device=sb.device)


def solver_mixed(solver) -> MixedPrecision:
    from ..precision import get_mixed

    return get_mixed(solver.prec.name, solver.cfg.rans.solve_precision)
