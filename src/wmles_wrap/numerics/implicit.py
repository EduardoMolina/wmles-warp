"""Steady implicit solver: backward Euler in pseudo-time with line relaxation (RANS core, R1).

Each outer iteration solves, approximately, the linear system of one backward-Euler step in local
pseudo-time,

    (V / dtau - dR/dQ) dQ = R(Q),        Q <- Q + alpha dQ,

with R = V dQ/dt the explicit residual in the residual precision (``numerics.precision``) and the
left-hand side in the solve precision (``rans.solve_precision``, float32 by default). Only the
convergence path depends on the left-hand side: at convergence R(Q) = 0 in the residual precision.

Approximate Jacobian (first order: Roe or Rusanov dissipation + thin-layer viscous). For cell i, face f with outward
area vector S and neighbour j:

    off-diagonal  O_f = 1/2 A(Q_j, S) - E_f - K_f dV/dU(Q_j)
    diagonal      D_i = lambda_i / CFL I + sum_f (E_f + K_f dV/dU(Q_i))
                        + sum_{wall/slip f} [1/2 A(Q_g, S) B + K_f (dV/dU(Q_i) - dV/dU(Q_g) B)]
                                                                     (those f not in the sum)
    E_f = eps c_i I (Rusanov, c_i = 1/2 max_{i,j}(|u.S| + a |S|)) or eps/2 |A_roe(S)| (Roe: the
          residual's Roe average and entropy fix, all eigenvalues floored at
          rans.jacobian_entropy_fix * a); eps = rans.implicit_dissipation (default 1),
          rans.jacobian (auto: Roe for a Roe residual, else Rusanov). Without the eigenvalue floor
          the Roe operator limit-cycles near 1e-7 on the laminar flat plate (|u.n| ~ 0 on faces
          along walls); a floor of 0.15-0.3 converges fastest (default 0.2), Rusanov slowest.
    K_f: thin-layer viscous flux F_v = K_f (V_j - V_i) in V = (rho, u, v, w, T):
         |S|^2 / V_f [mu (I + n n^T / 3) on u, kappa on T, work term u_f^T mu (I + n n^T / 3)]

A(Q, S) = d(F(Q).S)/dQ is the exact Euler flux Jacobian; sum_f A(Q_i, S_f) vanishes by the GCL,
so the cell's own central inviscid terms drop out. V/dtau = lambda_i / CFL uses the explicit time-step
spectral radius (local time stepping). Wall and slip faces use the linearized ghost state
(ghost dQ = B dQ_i) and no dissipation, as in the residual, so the operator conserves mass like
the residual (otherwise the corrections pump mass along the residual's null space). The viscous
blocks act on temperature, not on rho E: heat conduction couples the energy equation to rho and
rho u through T. Other physical BCs are explicit (ghost dQ = 0).

Relaxation along lines (the parent block's line axis, kept whole by the partitioner): one thread
per line runs a block-tridiagonal Thomas solve with the stored 5x5 line blocks; the four
off-line neighbours, and line neighbours across block interfaces, enter the right-hand side
through matrix-free products O_f dQ_j with the latest available dQ_j. ``zebra`` colours lines by
the parity of their parent-block off-line indices (red-black line Gauss-Seidel, in place);
``line_jacobi`` uses the previous sweep everywhere. dQ ghosts across interfaces are exchanged
after every colour/sweep, so the iteration does not depend on the partitioning.

The CFL follows switched evolution relaxation (SER), rounded to 4 significant digits so that the
round-off of the global norm (summation order) cannot make runs partitioning-dependent.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

import logging
import math
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from ..grid.block import face_axis
from ..precision import MixedPrecision, get_mixed, kernel_factory
from .integrator import bc_code
from .kernels import (BC_INFLOW_TOTAL, BC_OUTFLOW, BC_SLIP, BC_WALL_ADIABATIC, BC_WALL_ISOTHERMAL,
                      make_kernels)
from .mixed import block_kernels, mixed_kernels

log = logging.getLogger("wmles_wrap")


@kernel_factory
def implicit_kernels(mp: MixedPrecision):
    R = mp.residual
    rs, rv3, rv5 = R.scalar, R.vec3, R.vec5
    s, v3, v5, m55 = mp.solve.scalar, mp.solve.vec3, mp.solve.vec5, mp.solve.mat55
    v4 = wp.types.vector(4, s)
    Params = make_kernels(R)["Params"]
    _, _, block_inverse, _ = block_kernels(mp.solve)
    zero, one, half, two = s(0.0), s(1.0), s(0.5), s(2.0)
    four_thirds, c15, third = s(4.0 / 3.0), s(1.5), s(1.0 / 3.0)
    r_one = rs(1.0)
    r_half = rs(0.5)
    wall_a, wall_i, slip = BC_WALL_ADIABATIC, BC_WALL_ISOTHERMAL, BC_SLIP
    outflow, inflow_t = BC_OUTFLOW, BC_INFLOW_TOTAL

    @wp.func
    def to_s5(W: rv5):
        return v5(s(W[0]), s(W[1]), s(W[2]), s(W[3]), s(W[4]))

    @wp.func
    def to_s3(x: rv3):
        return v3(s(x[0]), s(x[1]), s(x[2]))

    @wp.func
    def visc(T: s, p: Params):
        m = one
        if p.visc_model == 0:
            sn = s(p.s_nd)
            m = wp.pow(T, c15) * (one + sn) / (T + sn)
        elif p.visc_model == 1:
            m = wp.pow(T, s(p.power_exp))
        return m * s(p.inv_re)

    @wp.func
    def jac_vec(W: v5, S: v3, d: v5, g: s):
        """A(Q, S) d for the Euler flux F(Q).S, Q from primitives W = (rho, u, v, w, p)."""
        rho = W[0]
        u = v3(W[1], W[2], W[3])
        g1 = g - one
        q2 = wp.dot(u, u)
        phi = half * g1 * q2
        H = g / g1 * W[4] / rho + half * q2
        Vn = wp.dot(u, S)
        dm = v3(d[1], d[2], d[3])
        Sdm = wp.dot(S, dm)
        udm = wp.dot(u, dm)
        rm = (phi * S - u * Vn) * d[0] + u * Sdm - (g1 * udm) * S + Vn * dm + (g1 * d[4]) * S
        r4 = Vn * (phi - H) * d[0] + H * Sdm - g1 * Vn * udm + g * Vn * d[4]
        return v5(Sdm, rm[0], rm[1], rm[2], r4)

    @wp.func
    def jac_mat(W: v5, S: v3, g: s):
        M = m55()
        for k in range(5):
            e = v5()
            e[k] = one
            col = jac_vec(W, S, e, g)
            for r in range(5):
                M[r, k] = col[r]
        return M

    @wp.func
    def face_coef(Wi: v5, Wj: v5, S: v3, p: Params):
        """Rusanov coefficient c_i = 1/2 max(|u.S| + a|S|) over both sides."""
        g = s(p.gamma)
        area = wp.length(S)
        ui = v3(Wi[1], Wi[2], Wi[3])
        uj = v3(Wj[1], Wj[2], Wj[3])
        ci = wp.sqrt(g * Wi[4] / Wi[0])
        cj = wp.sqrt(g * Wj[4] / Wj[0])
        return half * wp.max(wp.abs(wp.dot(ui, S)) + ci * area, wp.abs(wp.dot(uj, S)) + cj * area)

    @wp.func
    def harten(lam: s, delta: s):
        l = wp.abs(lam)
        if l < delta:
            l = (l * l + delta * delta) / (two * delta)
        return l

    @wp.func
    def roe_absA_vec(WL: v5, WR: v5, S: v3, d: v5, fix: s, fix_lin: s, p: Params):
        """|A_roe(S)| d: the residual's Roe dissipation (numerics/kernels.py roe_dissipation,
        same Roe average and Harten entropy fix) applied to the primitive increments of d."""
        g = s(p.gamma)
        g1 = g - one
        area = wp.length(S)
        n = S / area
        uL = v3(WL[1], WL[2], WL[3])
        uR = v3(WR[1], WR[2], WR[3])
        HL = g / g1 * WL[4] / WL[0] + half * wp.dot(uL, uL)
        HR = g / g1 * WR[4] / WR[0] + half * wp.dot(uR, uR)
        sL = wp.sqrt(WL[0])
        sR = wp.sqrt(WR[0])
        rt = sL * sR
        ut = (sL * uL + sR * uR) / (sL + sR)
        Ht = (sL * HL + sR * HR) / (sL + sR)
        q2 = wp.dot(ut, ut)
        at2 = wp.max(g1 * (Ht - half * q2), s(1e-30))
        at = wp.sqrt(at2)
        # primitive increments at the Roe state
        dm = v3(d[1], d[2], d[3])
        drho = d[0]
        du = (dm - ut * drho) / rt
        dp = g1 * (half * q2 * drho - wp.dot(ut, dm) + d[4])
        unt = wp.dot(ut, n)
        dun = wp.dot(du, n)
        delta = wp.max(s(p.entropy_fix), fix) * at
        l1 = harten(unt - at, delta)
        l2 = wp.abs(unt)
        if fix_lin > zero:
            l2 = harten(unt, fix_lin * at)
        l5 = harten(unt + at, delta)
        a1 = (dp - rt * at * dun) / (two * at2)
        a5 = (dp + rt * at * dun) / (two * at2)
        a2 = drho - dp / at2
        dut = du - dun * n
        d0 = l1 * a1 + l2 * a2 + l5 * a5
        dmo = l1 * a1 * (ut - at * n) + l2 * (a2 * ut + rt * dut) + l5 * a5 * (ut + at * n)
        de = (l1 * a1 * (Ht - at * unt) + l2 * (a2 * half * q2 + rt * (wp.dot(ut, du) - unt * dun))
              + l5 * a5 * (Ht + at * unt))
        return area * v5(d0, dmo[0], dmo[1], dmo[2], de)

    @wp.func
    def diss_vec(Wi: v5, Wj: v5, S: v3, d: v5, jc: v4, p: Params):
        """Implicit dissipation applied to d, jc = (type, eps, fix, fix_linear): eps/2 |A_roe| d with
        the acoustic / entropy-shear eigenvalues floored at fix * a / fix_linear * a (type 1), or
        eps c_i d (Rusanov, type 0)."""
        out = v5()
        if jc[0] > half:
            out = (half * jc[1]) * roe_absA_vec(Wi, Wj, S, d, jc[2], jc[3], p)
        else:
            out = (jc[1] * face_coef(Wi, Wj, S, p)) * d
        return out

    @wp.func
    def diss_mat(Wi: v5, Wj: v5, S: v3, jc: v4, p: Params):
        M = m55()
        for k in range(5):
            e = v5()
            e[k] = one
            col = diss_vec(Wi, Wj, S, e, jc, p)
            for r in range(5):
                M[r, k] = col[r]
        return M

    @wp.func
    def dV_dU(W: v5, p: Params):
        """d(rho, u, v, w, T)/d(conservative) at W = (rho, u, v, w, p)."""
        g1 = s(p.gamma) - one
        Rg = s(p.R)
        rho = W[0]
        T = W[4] / (rho * Rg)
        M = m55()
        M[0, 0] = one
        q2 = W[1] * W[1] + W[2] * W[2] + W[3] * W[3]
        M[4, 0] = (half * g1 * q2 - Rg * T) / (rho * Rg)
        for r in range(3):
            M[1 + r, 0] = -W[1 + r] / rho
            M[1 + r, 1 + r] = one / rho
            M[4, 1 + r] = -g1 * W[1 + r] / (rho * Rg)
        M[4, 4] = g1 / (rho * Rg)
        return M

    @wp.func
    def visc_K(Wi: v5, Wj: v5, S: v3, Vi: s, Vj: s, mut_f: s, p: Params):
        """Thin-layer viscous flux F_v = K (V_j - V_i), V = (rho, u, v, w, T):
        K = |S|^2 / V_f [mu (I + n n^T / 3) on u, kappa on T, work term u_f^T mu (I + n n^T / 3)]."""
        K = m55()
        if p.viscous == 1:
            Rg = s(p.R)
            Tf = half * (Wi[4] / (Wi[0] * Rg) + Wj[4] / (Wj[0] * Rg))
            mu = visc(Tf, p)
            mue = mu + mut_f
            kap = s(p.cp) * (mu / s(p.prandtl) + mut_f / s(p.prandtl_t))
            f = wp.dot(S, S) / (half * (Vi + Vj))
            n = S / wp.length(S)
            uf = half * v3(Wi[1] + Wj[1], Wi[2] + Wj[2], Wi[3] + Wj[3])
            for r in range(3):
                for c in range(3):
                    k = mue * n[r] * n[c] * third
                    if r == c:
                        k = k + mue
                    K[1 + r, 1 + c] = f * k
            for c in range(3):
                K[4, 1 + c] = uf[0] * K[1, 1 + c] + uf[1] * K[2, 1 + c] + uf[2] * K[3, 1 + c]
            K[4, 4] = f * kap
        return K

    @wp.func
    def off_mat(Wi: v5, Wj: v5, S: v3, K: m55, jc: v4, p: Params):
        """O = dRes_i/dU_j = 1/2 A(Q_j, S) - (dissipation) - K dV/dU(Q_j)."""
        return half * jac_mat(Wj, S, s(p.gamma)) - diss_mat(Wi, Wj, S, jc, p) - K * dV_dU(Wj, p)

    @wp.func
    def off_vec(Wi: v5, Wj: v5, S: v3, K: m55, d: v5, jc: v4, p: Params):
        """O d without forming O."""
        return half * jac_vec(Wj, S, d, s(p.gamma)) - diss_vec(Wi, Wj, S, d, jc, p) - K * (dV_dU(Wj, p) * d)

    @wp.func
    def face_S(Si: wp.array3d(dtype=rv3), Sj: wp.array3d(dtype=rv3), Sk: wp.array3d(dtype=rv3),
               idx: wp.vec3i, a: int, side: int):
        """Outward area vector of face (a, side) of cell idx (solve precision)."""
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
        out = to_s3(S)
        if side == 0:
            out = -out
        return out

    @wp.func
    def dU_dW(W: v5, g: s):
        """d(conservative)/d(primitive) at W = (rho, u, v, w, p)."""
        M = m55()
        M[0, 0] = one
        M[4, 0] = half * (W[1] * W[1] + W[2] * W[2] + W[3] * W[3])
        for r in range(3):
            M[1 + r, 0] = W[1 + r]
            M[1 + r, 1 + r] = W[0]
            M[4, 1 + r] = W[0] * W[1 + r]
        M[4, 4] = one / (g - one)
        return M

    @wp.func
    def dW_dU(W: v5, g: s):
        """d(primitive)/d(conservative) at W = (rho, u, v, w, p)."""
        M = m55()
        g1 = g - one
        M[0, 0] = one
        M[4, 0] = half * g1 * (W[1] * W[1] + W[2] * W[2] + W[3] * W[3])
        for r in range(3):
            M[1 + r, 0] = -W[1 + r] / W[0]
            M[1 + r, 1 + r] = one / W[0]
            M[4, 1 + r] = -g1 * W[1 + r]
        M[4, 4] = g1
        return M

    @wp.func
    def mirror(code: int, S: v3, Wi: v5, Wg: v5, p: Params):
        """B with ghost dQ = B dQ_i, the linearization of the BC ghost state (numerics/kernels.py
        k_bc): walls u_g = 2 u_w - u_i, p_g = p_i, T_g = T_i (adiabatic) or 2 T_w - T_i
        (isothermal), rho_g = p_g / (R T_g); slip reverses the normal velocity."""
        g = s(p.gamma)
        Mw = wp.identity(n=5, dtype=s)
        if code == wall_a or code == wall_i:
            Mw[1, 1] = -one
            Mw[2, 2] = -one
            Mw[3, 3] = -one
            if code == wall_i:
                Rg = s(p.R)
                Ti = Wi[4] / (Wi[0] * Rg)
                Tg = Wg[4] / (Wg[0] * Rg)
                # rho_g = p_i / (R T_g), T_g = 2 T_w - T_i, T_i = p_i / (rho_i R)
                a = Wg[4] / (Rg * Tg * Tg)
                Mw[0, 0] = -a * Ti / Wi[0]
                Mw[0, 4] = one / (Rg * Tg) + a / (Wi[0] * Rg)
        elif code == slip:
            n = S / wp.length(S)
            for r in range(3):
                for c in range(3):
                    Mw[1 + r, 1 + c] = Mw[1 + r, 1 + c] - two * n[r] * n[c]
        return dU_dW(Wg, g) * Mw * dW_dU(Wi, g)

    @wp.func
    def open_ghost(code: int, S: v3, Wi: v5, Wg: v5, p: Params):
        """B with ghost dQ = B dQ_i for the outflow and inflow_total ghosts (k_bc), S outward.
        Outflow: rho, u from the interior, p fixed (subsonic). inflow_total: the ghost is a function
        of R+ = u_n + 2a/(gamma-1) of the first cell only, B = dW_g/dR+ (x) dR+/dW_i."""
        g = s(p.gamma)
        g1 = g - one
        Rg = s(p.R)
        n = S / wp.length(S)
        Mw = wp.identity(n=5, dtype=s)
        if code == outflow:
            if wp.dot(v3(Wi[1], Wi[2], Wi[3]), n) < wp.sqrt(g * Wi[4] / Wi[0]):
                Mw[4, 4] = zero
        else:
            cp = g * Rg / g1
            Tb = Wg[4] / (Wg[0] * Rg)
            ug = v3(Wg[1], Wg[2], Wg[3])
            q = wp.length(ug)
            d = ug / wp.max(q, s(1e-30))
            T0 = Tb + half * q * q / cp
            c = wp.min(wp.dot(d, n), -s(0.1))
            a_i = wp.sqrt(g * Wi[4] / Wi[0])
            rp = wp.dot(v3(Wi[1], Wi[2], Wi[3]), n) + two * a_i / g1
            qa = c * c + two / g1
            qc = half * g1 * rp * rp - g * Rg * T0 * c * c
            sq = wp.sqrt(wp.max(rp * rp - qa * qc, s(1e-30)))
            da = (one + rp * (one - half * qa * g1) / sq) / qa
            dq = (one - two * da / g1) / c
            dT = -q * dq / cp
            dp = Wg[4] * g / (g1 * Tb) * dT
            col = v5(dp / (Rg * Tb) - Wg[0] * dT / Tb, dq * d[0], dq * d[1], dq * d[2], dp)
            row = v5(-a_i / (g1 * Wi[0]), n[0], n[1], n[2], a_i / (g1 * Wi[4]))
            Mw = wp.outer(col, row)
        return dU_dW(Wg, g) * Mw * dW_dU(Wi, g)

    @wp.func
    def cell(axis: int, n: int, t1: int, t2: int):
        if axis == 0:
            return wp.vec3i(n, t1, t2)
        if axis == 1:
            return wp.vec3i(t1, n, t2)
        return wp.vec3i(t1, t2, n)

    @wp.kernel(module="unique")
    def k_assemble(
        W: wp.array3d(dtype=rv5),
        mut: wp.array3d(dtype=rs),
        Si: wp.array3d(dtype=rv3),
        Sj: wp.array3d(dtype=rv3),
        Sk: wp.array3d(dtype=rv3),
        vol: wp.array3d(dtype=rs),
        cfl: s,
        ng: int,
        nc: wp.vec3i,
        la: int,
        fcode: wp.array(dtype=int),
        p: Params,
        jc: v4,
        D: wp.array3d(dtype=m55),
        Lo: wp.array3d(dtype=m55),
        Up: wp.array3d(dtype=m55),
    ):
        a0, b0, c0 = wp.tid()
        idx = wp.vec3i(a0 + ng, b0 + ng, c0 + ng)
        Wi = to_s5(W[idx[0], idx[1], idx[2]])
        Vi = s(vol[idx[0], idx[1], idx[2]])
        g = s(p.gamma)
        ui = v3(Wi[1], Wi[2], Wi[3])
        snd = wp.sqrt(g * Wi[4] / Wi[0])
        # local pseudo-time step: V / dtau = lambda / CFL (spectral radius of the explicit solver)
        lam = s(0.0)
        ssum = s(0.0)
        for a in range(3):
            sa = half * (face_S(Si, Sj, Sk, idx, a, 1) - face_S(Si, Sj, Sk, idx, a, 0))
            lam = lam + wp.abs(wp.dot(ui, sa)) + snd * wp.length(sa)
            ssum = ssum + wp.dot(sa, sa)
        if p.viscous == 1:
            mu = visc(Wi[4] / (Wi[0] * s(p.R)), p) + s(mut[idx[0], idx[1], idx[2]])
            lam = lam + two * mu / Wi[0] * wp.max(four_thirds, g / s(p.prandtl)) * ssum / Vi
        dsc = lam / cfl
        Dm = m55()
        lo_b = m55()
        up_b = m55()
        for a in range(3):
            for side in range(2):
                nb = wp.vec3i(idx[0], idx[1], idx[2])
                if side == 0:
                    nb[a] = nb[a] - 1
                else:
                    nb[a] = nb[a] + 1
                S = face_S(Si, Sj, Sk, idx, a, side)
                Wj = to_s5(W[nb[0], nb[1], nb[2]])
                Vj = s(vol[nb[0], nb[1], nb[2]])
                mf = half * s(mut[idx[0], idx[1], idx[2]] + mut[nb[0], nb[1], nb[2]])
                K = visc_K(Wi, Wj, S, Vi, Vj, mf, p)
                inside = nb[a] >= ng and nb[a] < ng + nc[a]
                code = fcode[2 * a + side]
                if (not inside) and (code == wall_a or code == wall_i or code == slip):
                    # wall/slip: exact central inviscid part (no dissipation in the residual there)
                    # and the viscous part, with the linearized ghost state dQ_g = B dQ_i
                    B = mirror(code, S, Wi, Wj, p)
                    Dm = Dm + half * jac_mat(Wj, S, g) * B + K * (dV_dU(Wi, p) - dV_dU(Wj, p) * B)
                else:
                    Dm = Dm + diss_mat(Wi, Wj, S, jc, p) + K * dV_dU(Wi, p)
                    if (not inside) and (code == outflow or code == inflow_t):
                        # linearized open-boundary ghost (the far field stays frozen)
                        Dm = Dm + off_mat(Wi, Wj, S, K, jc, p) * open_ghost(code, S, Wi, Wj, p)
                    if a == la and inside:
                        if side == 0:
                            lo_b = off_mat(Wi, Wj, S, K, jc, p)
                        else:
                            up_b = off_mat(Wi, Wj, S, K, jc, p)
        D[idx[0], idx[1], idx[2]] = Dm + dsc * wp.identity(n=5, dtype=s)
        Lo[idx[0], idx[1], idx[2]] = lo_b
        Up[idx[0], idx[1], idx[2]] = up_b

    @wp.kernel(module="unique")
    def k_line(
        W: wp.array3d(dtype=rv5),
        mut: wp.array3d(dtype=rs),
        Si: wp.array3d(dtype=rv3),
        Sj: wp.array3d(dtype=rv3),
        Sk: wp.array3d(dtype=rv3),
        vol: wp.array3d(dtype=rs),
        ng: int,
        nc: wp.vec3i,
        la: int,
        fcode: wp.array(dtype=int),
        p: Params,
        jc: v4,
        D: wp.array3d(dtype=m55),
        Lo: wp.array3d(dtype=m55),
        Up: wp.array3d(dtype=m55),
        rhs: wp.array3d(dtype=v5),
        dq_in: wp.array3d(dtype=v5),
        color: int,
        offset: int,
        G: wp.array3d(dtype=m55),
        Y: wp.array3d(dtype=v5),
        dq_out: wp.array3d(dtype=v5),
    ):
        t1, t2 = wp.tid()
        if color >= 0 and (offset + t1 + t2) % 2 != color:
            return
        g = s(p.gamma)
        n = nc[la]
        Gp = m55()
        yp = v5()
        for c in range(n):
            idx = cell(la, c + ng, t1 + ng, t2 + ng)
            r = rhs[idx[0], idx[1], idx[2]]
            Wi = to_s5(W[idx[0], idx[1], idx[2]])
            Vi = s(vol[idx[0], idx[1], idx[2]])
            for a in range(3):
                for side in range(2):
                    nb = wp.vec3i(idx[0], idx[1], idx[2])
                    if side == 0:
                        nb[a] = nb[a] - 1
                    else:
                        nb[a] = nb[a] + 1
                    inside = nb[a] >= ng and nb[a] < ng + nc[a]
                    code = fcode[2 * a + side]
                    # skip line neighbours inside the block (in the line solve) and physical boundaries
                    # (folded into D or explicit)
                    use = (not (a == la and inside)) and (inside or code < 0)
                    if use:
                        S = face_S(Si, Sj, Sk, idx, a, side)
                        Wj = to_s5(W[nb[0], nb[1], nb[2]])
                        Vj = s(vol[nb[0], nb[1], nb[2]])
                        mf = half * s(mut[idx[0], idx[1], idx[2]] + mut[nb[0], nb[1], nb[2]])
                        K = visc_K(Wi, Wj, S, Vi, Vj, mf, p)
                        r = r - off_vec(Wi, Wj, S, K, dq_in[nb[0], nb[1], nb[2]], jc, p)
            L = Lo[idx[0], idx[1], idx[2]]
            Dc = D[idx[0], idx[1], idx[2]]
            if c > 0:
                Dc = Dc - L * Gp
                r = r - L * yp
            inv = block_inverse(Dc)
            Gp = inv * Up[idx[0], idx[1], idx[2]]
            yp = inv * r
            G[idx[0], idx[1], idx[2]] = Gp
            Y[idx[0], idx[1], idx[2]] = yp
        x = v5(yp)
        last = cell(la, n - 1 + ng, t1 + ng, t2 + ng)
        dq_out[last[0], last[1], last[2]] = x
        for cc in range(n - 1):
            c = n - 2 - cc
            idx = cell(la, c + ng, t1 + ng, t2 + ng)
            x = Y[idx[0], idx[1], idx[2]] - G[idx[0], idx[1], idx[2]] * x
            dq_out[idx[0], idx[1], idx[2]] = x

    @wp.kernel(module="unique")
    def k_update(dq: wp.array3d(dtype=v5), ng: int, limit: rs, p: Params, U: wp.array3d(dtype=rv5)):
        """U += alpha dQ, alpha <= 1 limiting the relative density and pressure change."""
        a0, b0, c0 = wp.tid()
        i = a0 + ng
        j = b0 + ng
        k = c0 + ng
        Uc = U[i, j, k]
        d5 = dq[i, j, k]
        d = rv5(rs(d5[0]), rs(d5[1]), rs(d5[2]), rs(d5[3]), rs(d5[4]))
        g1 = p.gamma - r_one
        p0 = g1 * (Uc[4] - r_half * (Uc[1] * Uc[1] + Uc[2] * Uc[2] + Uc[3] * Uc[3]) / Uc[0])
        alpha = rs(1.0)
        if wp.abs(d[0]) > limit * Uc[0]:
            alpha = limit * Uc[0] / wp.abs(d[0])
        for it in range(30):
            Un = Uc + alpha * d
            pn = g1 * (Un[4] - r_half * (Un[1] * Un[1] + Un[2] * Un[2] + Un[3] * Un[3]) / Un[0])
            if wp.abs(pn - p0) <= limit * p0:
                break
            alpha = r_half * alpha
        U[i, j, k] = Uc + alpha * d

    return k_assemble, k_line, k_update


def sweep_converged(pairs, ng: int, tol: float) -> bool:
    """max|new - old| <= tol max|new| over all interior cells (max norms do not depend on the
    summation order, so the sweep count is the same for every partitioning)."""
    num = den = 0.0
    for new, old in pairs:
        a = new.numpy()[ng:-ng, ng:-ng, ng:-ng]
        b = old.numpy()[ng:-ng, ng:-ng, ng:-ng]
        num = max(num, float(np.abs(a - b).max()))
        den = max(den, float(np.abs(a).max()))
    return num <= tol * den


def jacobian_type(cfg) -> str:
    """rans.jacobian with auto resolved: roe for a Roe residual, else rusanov."""
    if cfg.rans.jacobian != "auto":
        return cfg.rans.jacobian
    num = cfg.numerics
    return "roe" if num.flux == "roe_muscl" or num.dissipation == "roe" else "rusanov"


@dataclass
class LineLayout:
    la: int  # line axis
    dims: tuple  # (n1, n2) interior cells of the off-line axes
    offset: int  # parity offset: parent index of the first line's off-line coordinates
    fcode: wp.array  # (6,) BC code per face, -1 for interfaces/partition faces


def line_layout(solver) -> dict[int, LineLayout]:
    """Line axis, line grid, colour parity offset and face BC codes of every sub-block."""
    if solver.line_axes is None:
        raise RuntimeError("the implicit solver needs line axes (parallel.keep_lines)")
    out = {}
    for sb in solver.subs:
        la = solver.line_axes[sb.parent]
        if sb.lo[la] != 0 or sb.hi[la] != sb.parent_nn[la] - 1:
            raise RuntimeError(f"sub-block {sb.id} does not contain whole lines along axis {la}")
        o1, o2 = [a for a in range(3) if a != la]
        codes = [bc_code(sb.face_bc[f], "none") if f in sb.face_bc else -1 for f in range(6)]
        out[sb.id] = LineLayout(la, (sb.nc[o1], sb.nc[o2]), (sb.lo[o1] + sb.lo[o2]) % 2,
                                wp.array(np.array(codes, dtype=np.int32), dtype=int, device=sb.device))
    return out


@dataclass
class _Lines:
    la: int
    dims: tuple
    offset: int
    fcode: wp.array
    D: wp.array
    Lo: wp.array
    Up: wp.array
    G: wp.array
    Y: wp.array
    rhs: wp.array
    dq: wp.array
    dq2: wp.array


class ImplicitSolver:
    """Outer pseudo-time iterations for ``solver.mode: rans`` (see module docstring)."""

    def __init__(self, solver):
        self.s = solver
        self.cfg = solver.cfg.rans
        self.mp = get_mixed(solver.prec.name, self.cfg.solve_precision)
        self.k_assemble, self.k_line, self.k_update = implicit_kernels(self.mp)
        self.k_rhs, _ = mixed_kernels(self.mp)
        self.I = solver.integrator
        self.jacobian = jacobian_type(solver.cfg)
        self.jc = wp.types.vector(4, self.mp.solve.scalar)(
            1.0 if self.jacobian == "roe" else 0.0, self.cfg.implicit_dissipation, self.cfg.jacobian_entropy_fix,
            self.cfg.jacobian_entropy_fix_linear)
        self.iteration = 0
        self.last_sweeps = 0
        self.res0 = None
        self.history: list[tuple[int, float, float, np.ndarray]] = []
        sp = self.mp.solve
        ng = solver.ng
        self.lines: dict[int, _Lines] = {}
        for sid, lay in line_layout(solver).items():
            sb = next(b for b in solver.subs if b.id == sid)
            z = lambda dt: wp.zeros(sb.padded_shape_cells(ng), dtype=dt, device=sb.device)  # noqa: E731
            self.lines[sid] = _Lines(
                la=lay.la, dims=lay.dims, offset=lay.offset, fcode=lay.fcode,
                D=z(sp.mat55), Lo=z(sp.mat55), Up=z(sp.mat55), G=z(sp.mat55), Y=z(sp.vec5),
                rhs=z(sp.vec5), dq=z(sp.vec5), dq2=z(sp.vec5))
        self.turb = solver.turbulence
        self.res0_turb = None
        st = self.I.params.w_inf
        rho, u, v, w, pr = (float(x) for x in st)
        self.ref = np.array([rho, rho, rho, rho, pr / (solver.gas.gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w)])

    # ---- one outer iteration ---------------------------------------------------------------
    def residual_norm(self) -> tuple[float, np.ndarray]:
        """Scaled RMS of R/V over all interior cells: (combined, per variable)."""
        per = self.I.residual_norms()
        return float(np.sqrt(np.sum((per / self.ref) ** 2))), per

    def cfl(self, res: float) -> float:
        c = self.cfg
        if self.res0 is None or res <= 0:
            val = c.cfl_start
        else:
            val = min(c.cfl_max, max(c.cfl_start, c.cfl_start * (self.res0 / res) ** c.ser_exponent))
        return float(f"{val:.4g}")

    def assemble(self, cfl: float) -> None:
        """Jacobian blocks and rhs = R from the primitives/residual of the last ``Integrator.rhs``."""
        s, ng, P = self.s, self.s.ng, self.I.params
        sc = self.mp.solve.scalar
        for sid, d in self.I.data.items():
            L, m, dev = self.lines[sid], s.metrics[sid], d.sub.device
            wp.launch(self.k_assemble, dim=d.sub.nc,
                      inputs=[d.W, d.mut, m.Si, m.Sj, m.Sk, m.vol, sc(cfl), ng, wp.vec3i(*d.sub.nc), L.la, L.fcode, P,
                              self.jc],
                      outputs=[L.D, L.Lo, L.Up], device=dev)
            wp.launch(self.k_rhs, dim=d.sub.nc, inputs=[d.R, ng], outputs=[L.rhs], device=dev)
            L.dq.zero_()
            L.dq2.zero_()

    def relax(self, sweeps: int | None = None) -> int:
        """Line relaxation sweeps on the assembled system; the correction ends up in ``dq``. With
        ``sweeps`` None: adaptive (rans.min_sweeps .. rans.sweeps, rans.sweep_tolerance). Returns
        the number of sweeps."""
        s, ng, P = self.s, self.s.ng, self.I.params
        zebra = self.cfg.smoother == "zebra"
        fixed = sweeps is not None
        nmax = sweeps if fixed else self.cfg.sweeps
        done = 0
        for sweep in range(nmax):
            check = (not fixed) and sweep + 1 >= self.cfg.min_sweeps
            if check and zebra:
                for L in self.lines.values():
                    wp.copy(L.dq2, L.dq)
            for color in ((0, 1) if zebra else (-1,)):
                for sid, d in self.I.data.items():
                    L, m, dev = self.lines[sid], s.metrics[sid], d.sub.device
                    out = L.dq if zebra else L.dq2
                    wp.launch(self.k_line, dim=L.dims,
                              inputs=[d.W, d.mut, m.Si, m.Sj, m.Sk, m.vol, ng, wp.vec3i(*d.sub.nc), L.la, L.fcode,
                                      P, self.jc, L.D, L.Lo, L.Up,
                                      L.rhs, L.dq, color, L.offset, L.G, L.Y],
                              outputs=[out], device=dev)
                if not zebra:
                    for L in self.lines.values():
                        L.dq, L.dq2 = L.dq2, L.dq
                s.halo.exchange({sid: L.dq for sid, L in self.lines.items()}, stages=(0,))
            done = sweep + 1
            if check and sweep_converged([(L.dq, L.dq2) for L in self.lines.values()], ng, self.cfg.sweep_tolerance):
                break
        self.last_sweeps = done
        return done

    def update(self) -> None:
        s, ng = self.s, self.s.ng
        lim = self.mp.residual.scalar(self.cfg.relax_limit)
        for sid, d in self.I.data.items():
            wp.launch(self.k_update, dim=d.sub.nc, inputs=[self.lines[sid].dq, ng, lim, self.I.params],
                      outputs=[s.U[sid]], device=d.sub.device)

    def iterate(self) -> float:
        """Residual, CFL, assembly, line relaxation and update. Returns the residual norm of the
        state before the update."""
        self.I.rhs(0.0, 0)
        res, per = self.residual_norm()
        rt = 0.0
        if self.turb is not None:
            self.turb.residual()
            rt = self.turb.residual_norm()
            if self.res0_turb is None:
                self.res0_turb = rt if rt > 0 else 1.0
        if self.res0 is None:
            self.res0 = res if res > 0 else 1.0
        cfl = self.cfl(res)
        self.assemble(cfl)
        self.relax()
        self.update()
        if self.turb is not None:
            self.turb.implicit_step(cfl, self.cfg, self.cfg.smoother == "zebra")
        self.iteration += 1
        self.history.append((self.iteration, cfl, res, per, rt))
        return res

    def converged(self) -> bool:
        h = self.history[-1]
        ok = h[2] <= self.cfg.tolerance * self.res0
        if self.turb is not None:
            ok = ok and h[4] <= self.cfg.tolerance * self.res0_turb
        return ok

    def run(self, max_iterations: int | None = None, log_every: int = 10, history_file=None) -> bool:
        """Iterate until the residual has dropped by ``rans.tolerance`` (returns True) or the
        iteration limit is reached. The last norm is that of the final state."""
        n = self.cfg.max_iterations if max_iterations is None else max_iterations
        t0 = _time.perf_counter()
        converged = False
        for _ in range(n):
            res = self.iterate()
            rt = self.history[-1][4]
            if self.iteration % max(log_every, 1) == 0 or self.iteration == 1:
                extra = f"  nut drop {rt / self.res0_turb:.3e}" if self.turb is not None else ""
                log.info("iter %6d  cfl %9.4g  res %.6e  drop %.3e%s  (%.1f s)", self.iteration, self.history[-1][1],
                         res, res / self.res0, extra, _time.perf_counter() - t0)
            if not (math.isfinite(res) and math.isfinite(rt)):
                raise FloatingPointError(f"residual is {res} (nut {rt}) at iteration {self.iteration}")
            if self.converged():
                converged = True
                break
        if history_file is not None:
            Path(history_file).parent.mkdir(parents=True, exist_ok=True)
            self.write_history(history_file)
        return converged

    def write_history(self, path) -> None:
        lines = ["# iter cfl res(scaled) rms(R/V): rho rhou rhov rhow rhoE; rms(R_nut/V)/nut_inf"]
        for it, cfl, res, per, rt in self.history:
            lines.append(f"{it:7d} {cfl:11.4e} {res:.10e} " + " ".join(f"{x:.6e}" for x in per) + f" {rt:.6e}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
