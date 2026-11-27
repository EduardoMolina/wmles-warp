"""FGMRES for the pseudo-time linear system of the steady implicit solver (R4, docs/plan_rans.md).

Solves the backward-Euler system of one outer iteration of ``numerics/implicit.py``,
(V/dtau - dF/dX) dX = F(X), with right-preconditioned flexible GMRES (Saad 1993): one cycle of at
most ``rans.krylov_restart`` vectors, stopping at a relative drop ``rans.krylov_tolerance`` of
the linear residual.

- **Unknowns.** X = Q (mean flow) or, for ``jfnk`` with SA, X = (Q, nu tilde) **coupled**: the
  segregated SA step cannot follow Newton-size mean-flow corrections (TMR 2DZP stalled at a 2e-3
  drop), so nu tilde joins the Krylov vectors and its update is taken from the same solve.
- **Preconditioner:** the line relaxations of the approximate first-order operators, block
  diagonal in (Q, nu tilde): ``ImplicitSolver.relax`` and ``SAModel.relax`` from zero,
  ``rans.krylov_sweeps`` sweeps each (0 = their adaptive rule; the preconditioner then changes
  between applications, hence the flexible variant). It hardly couples cells across the lines, so
  slow modes along the other directions need many Krylov vectors. If a cycle reduces the linear
  residual by less than ``rans.krylov_stall`` (GMRES(m) stagnation), the plain preconditioner
  step (one defect-correction step) is taken instead.
- **Matrix-vector product** (``rans.krylov_matvec``):
  - ``jfnk``: Jacobian-free, J z = V/dtau z - (F(X + eps z) - F(X)) / eps with the full residuals
    (second-order MUSCL, boundary conditions, eddy viscosity, SA) in the residual precision. eps
    makes the largest change of a variable, relative to its scale, equal to ``rans.fd_epsilon``.
    This removes the operator mismatch that limits plain defect correction: the outer iteration
    becomes inexact Newton as the CFL grows.
  - ``operator``: the approximate mean-flow operator itself (``k_apply``); it only accelerates the
    inner solve (e.g. across cells that are thin across the lines); SA stays segregated.
- **Inner product:** <x, y> = sum_i sum_c x_ic y_ic / (V_i/dtau_i ref_c)^2, i.e. a residual is
  measured by the pseudo-time update dtau/V F it would cause, in units of the variable scales
  ref_c: the free-stream state for Q, max(max |nu tilde|, nu tilde_inf) for nu tilde. Each sum
  runs along whole lines on the device in float64 (the partitioner keeps lines whole) and over
  the lines on the host in parent-block order, so every dot product, and with it the iteration,
  is bitwise independent of the partitioning. Orthogonalization: classical Gram-Schmidt with one
  reorthogonalization (CGS2), two batched reductions per Krylov vector.
- **Storage:** 2 m + 1 vectors in the solve precision (float32 by default: 20 B per cell and
  vector, 24 B coupled), plus a copy of X and F(X) in the residual precision for ``jfnk``.

After a ``jfnk`` solve the state is restored exactly. The primitives, gradients and eddy viscosity
left in the integrator are those of the last perturbed residual evaluation (relative change ~
fd_epsilon); nothing after the solve reads them before the next residual evaluation, except the
segregated SA operator assembly with ``operator``, which does not perturb.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

import math

import numpy as np
import warp as wp

from ..precision import MixedPrecision, kernel_factory


@kernel_factory
def krylov_kernels(mp: MixedPrecision):
    rs, rv5 = mp.residual.scalar, mp.residual.vec5
    s, v5 = mp.solve.scalar, mp.solve.vec5
    f64 = wp.float64
    d5 = wp.types.vector(5, f64)

    @wp.func
    def cell(axis: int, n: int, t1: int, t2: int):
        if axis == 0:
            return wp.vec3i(n, t1, t2)
        if axis == 1:
            return wp.vec3i(t1, n, t2)
        return wp.vec3i(t1, t2, n)

    @wp.func
    def to_d5(x: v5):
        return d5(f64(x[0]), f64(x[1]), f64(x[2]), f64(x[3]), f64(x[4]))

    @wp.kernel(module="unique")
    def k_dots(V: wp.array4d(dtype=v5), w: wp.array3d(dtype=v5), Vt: wp.array4d(dtype=s), wt: wp.array3d(dtype=s),
               nvec: int, has_t: int, wc: wp.array3d(dtype=s), wv: d5, wnu: f64, ng: int, n: int, la: int,
               out: wp.array3d(dtype=f64)):
        """out[k, line] = weighted sum along the line of V[k] . w (+ Vt[k] wt), k < nvec."""
        t1, t2 = wp.tid()
        for k in range(nvec):
            acc = f64(0.0)
            for c in range(n):
                idx = cell(la, c + ng, t1 + ng, t2 + ng)
                a = to_d5(V[k, idx[0], idx[1], idx[2]])
                b = to_d5(w[idx[0], idx[1], idx[2]])
                q2 = f64(wc[idx[0], idx[1], idx[2]])
                q2 = q2 * q2
                for q in range(5):
                    acc = acc + a[q] * b[q] * q2 * wv[q] * wv[q]
                if has_t == 1:
                    acc = acc + f64(Vt[k, idx[0], idx[1], idx[2]]) * f64(wt[idx[0], idx[1], idx[2]]) * q2 * wnu * wnu
            out[k, t1, t2] = acc

    @wp.kernel(module="unique")
    def k_linemax(x: wp.array3d(dtype=v5), xt: wp.array3d(dtype=s), has_t: int, wv: d5, wnu: f64, ng: int, n: int,
                  la: int, out: wp.array2d(dtype=f64)):
        """out[line] = max along the line of |x_c| wv_c (and |xt| wnu)."""
        t1, t2 = wp.tid()
        m = f64(0.0)
        for c in range(n):
            idx = cell(la, c + ng, t1 + ng, t2 + ng)
            a = to_d5(x[idx[0], idx[1], idx[2]])
            for q in range(5):
                m = wp.max(m, wp.abs(a[q]) * wv[q])
            if has_t == 1:
                m = wp.max(m, wp.abs(f64(xt[idx[0], idx[1], idx[2]])) * wnu)
        out[t1, t2] = m

    @wp.kernel(module="unique")
    def k_combo(V: wp.array4d(dtype=v5), Vt: wp.array4d(dtype=s), coef: wp.array(dtype=f64), nvec: int,
                has_t: int, base: wp.array3d(dtype=v5), base_t: wp.array3d(dtype=s), alpha: f64, ng: int,
                out: wp.array3d(dtype=v5), out_t: wp.array3d(dtype=s)):
        """out = alpha base + sum_{k < nvec} coef[k] V[k] (interior; out may be base)."""
        a0, b0, c0 = wp.tid()
        i, j, k = a0 + ng, b0 + ng, c0 + ng
        acc = alpha * to_d5(base[i, j, k])
        for q in range(nvec):
            acc = acc + coef[q] * to_d5(V[q, i, j, k])
        out[i, j, k] = v5(s(acc[0]), s(acc[1]), s(acc[2]), s(acc[3]), s(acc[4]))
        if has_t == 1:
            at = alpha * f64(base_t[i, j, k])
            for q in range(nvec):
                at = at + coef[q] * f64(Vt[q, i, j, k])
            out_t[i, j, k] = s(at)

    @wp.kernel(module="unique")
    def k_perturb(U0: wp.array3d(dtype=rv5), z: wp.array3d(dtype=v5), eps: rs, ng: int, U: wp.array3d(dtype=rv5)):
        a0, b0, c0 = wp.tid()
        i, j, k = a0 + ng, b0 + ng, c0 + ng
        zz = z[i, j, k]
        U[i, j, k] = U0[i, j, k] + eps * rv5(rs(zz[0]), rs(zz[1]), rs(zz[2]), rs(zz[3]), rs(zz[4]))

    @wp.kernel(module="unique")
    def k_perturb_t(n0: wp.array3d(dtype=rs), z: wp.array3d(dtype=s), eps: rs, ng: int, nt: wp.array3d(dtype=rs)):
        a0, b0, c0 = wp.tid()
        i, j, k = a0 + ng, b0 + ng, c0 + ng
        nt[i, j, k] = n0[i, j, k] + eps * rs(z[i, j, k])

    @wp.kernel(module="unique")
    def k_jfnk(z: wp.array3d(dtype=v5), dts: wp.array3d(dtype=s), R: wp.array3d(dtype=rv5),
               R0: wp.array3d(dtype=rv5), inv_eps: rs, ng: int, w: wp.array3d(dtype=v5)):
        """w = V/dtau z - (R(X + eps z) - R(X)) / eps (R is in interior-cell indexing)."""
        a0, b0, c0 = wp.tid()
        i, j, k = a0 + ng, b0 + ng, c0 + ng
        dr = (R[a0, b0, c0] - R0[a0, b0, c0]) * inv_eps
        zz = z[i, j, k]
        dt = dts[i, j, k]
        w[i, j, k] = v5(dt * zz[0] - s(dr[0]), dt * zz[1] - s(dr[1]), dt * zz[2] - s(dr[2]),
                        dt * zz[3] - s(dr[3]), dt * zz[4] - s(dr[4]))

    @wp.kernel(module="unique")
    def k_jfnk_t(z: wp.array3d(dtype=s), dts: wp.array3d(dtype=s), R: wp.array3d(dtype=rs), R0: wp.array3d(dtype=rs),
                 inv_eps: rs, ng: int, w: wp.array3d(dtype=s)):
        a0, b0, c0 = wp.tid()
        i, j, k = a0 + ng, b0 + ng, c0 + ng
        w[i, j, k] = dts[i, j, k] * z[i, j, k] - s((R[a0, b0, c0] - R0[a0, b0, c0]) * inv_eps)

    @wp.kernel(module="unique")
    def k_inv(x: wp.array3d(dtype=s), ng: int, out: wp.array3d(dtype=s)):
        a0, b0, c0 = wp.tid()
        i, j, k = a0 + ng, b0 + ng, c0 + ng
        out[i, j, k] = s(1.0) / x[i, j, k]

    @wp.kernel(module="unique")
    def k_to_rhs(x: wp.array3d(dtype=s), ng: int, out: wp.array3d(dtype=rs)):
        """Padded solve-precision scalar -> interior residual-precision scalar (SA line rhs)."""
        a0, b0, c0 = wp.tid()
        out[a0, b0, c0] = rs(x[a0 + ng, b0 + ng, c0 + ng])

    @wp.kernel(module="unique")
    def k_from_rhs(x: wp.array3d(dtype=rs), ng: int, out: wp.array3d(dtype=s)):
        a0, b0, c0 = wp.tid()
        out[a0 + ng, b0 + ng, c0 + ng] = s(x[a0, b0, c0])

    return dict(dots=k_dots, from_rhs=k_from_rhs, linemax=k_linemax, combo=k_combo, perturb=k_perturb,
                perturb_t=k_perturb_t,
                jfnk=k_jfnk, jfnk_t=k_jfnk_t, inv=k_inv, to_rhs=k_to_rhs)


class FGMRES:
    """Krylov workspace and solver attached to an ``ImplicitSolver`` (see module docstring)."""

    def __init__(self, imp):
        self.imp = imp
        s = imp.s
        self.s, self.ng = s, s.ng
        cfg = imp.cfg
        self.m = m = cfg.krylov_restart
        self.jfnk = cfg.krylov_matvec == "jfnk"
        self.turb = imp.turb if self.jfnk else None  # coupled (Q, nut) unknowns
        self.coupled = self.turb is not None
        self.K = krylov_kernels(imp.mp)
        sp, rp = imp.mp.solve, imp.mp.residual
        f64 = wp.float64
        self.wv = wp.types.vector(5, f64)(*(1.0 / imp.ref))
        self.wnu = 1.0
        self.iterations = 0
        self.sweeps = 0
        self.drop = 1.0
        self.stalled = False
        self.ws = {}
        self.pbuf = {}  # parent-block buffers of per-line sums: (m + 1, N1, N2) over the off-line axes
        for sid, L in imp.lines.items():
            sb = next(b for b in s.subs if b.id == sid)
            dev, pad = sb.device, sb.padded_shape_cells(self.ng)
            z = lambda shape, dt: wp.zeros(shape, dtype=dt, device=dev)  # noqa: E731
            o1, o2 = [a for a in range(3) if a != L.la]
            w = dict(V=z((m + 1,) + pad, sp.vec5), Z=z((m,) + pad, sp.vec5), w=z(pad, sp.vec5),
                     wc=z(pad, sp.scalar), dots=z((m + 1,) + L.dims, f64), lmax=z(L.dims, f64),
                     coef=z(m + 1, f64), sl=(slice(sb.lo[o1], sb.lo[o1] + L.dims[0]),
                                             slice(sb.lo[o2], sb.lo[o2] + L.dims[1])),
                     parent=sb.parent, n=sb.nc[L.la], nc=sb.nc, dev=dev)
            if self.coupled:
                w.update(Vt=z((m + 1,) + pad, sp.scalar), Zt=z((m,) + pad, sp.scalar), wt=z(pad, sp.scalar),
                         n0=z(pad, rp.scalar), Rt0=z(sb.nc, rp.scalar), rhs_t=z(sb.nc, rp.scalar))
            else:  # unused placeholders for the optional nut arguments
                w.update(Vt=z((m + 1, 1, 1, 1), sp.scalar), Zt=z((m, 1, 1, 1), sp.scalar), wt=z((1, 1, 1), sp.scalar))
            if self.jfnk:
                w["U0"] = z(pad, rp.vec5)
                w["R0"] = z(sb.nc, rp.vec5)
            self.ws[sid] = w
            if sb.parent not in self.pbuf:
                nn = sb.parent_nn
                self.pbuf[sb.parent] = np.zeros((m + 1, nn[o1] - 1, nn[o2] - 1))

    # ---- vectors: (key of the mean-flow part, key of the nut part, index or None) ----------------
    def _get(self, w, vec):
        key, key_t, j = vec
        x = w[key] if j is None else w[key][j]
        if not self.coupled:
            return x, w["wt"]  # placeholder
        return x, (w[key_t] if j is None else w[key_t][j])

    # ---- deterministic reductions ---------------------------------------------------------
    def _gather(self, key: str, nvec: int) -> np.ndarray:
        for w in self.ws.values():
            self.pbuf[w["parent"]][(slice(0, nvec),) + w["sl"]] = w[key].numpy()[:nvec]
        tot = np.zeros(nvec)
        for p in sorted(self.pbuf):
            tot += self.pbuf[p][:nvec].sum(axis=(1, 2))
        return tot

    def dots(self, nvec: int, vec, first: int = 0) -> np.ndarray:
        """<V[first + k], vec> for k < nvec (weighted inner product, partition-independent sums)."""
        L = self.imp.lines
        for sid, w in self.ws.items():
            x, xt = self._get(w, vec)
            Vt = w["Vt"][first:first + nvec] if self.coupled else w["Vt"]
            wp.launch(self.K["dots"], dim=L[sid].dims,
                      inputs=[w["V"][first:first + nvec], x, Vt, xt, nvec, int(self.coupled), w["wc"], self.wv,
                              self.wnu, self.ng, w["n"], L[sid].la],
                      outputs=[w["dots"]], device=w["dev"])
        return self._gather("dots", nvec)

    def norm_v(self, j: int) -> float:
        return math.sqrt(max(float(self.dots(1, ("V", "Vt", j), first=j)[0]), 0.0))

    def maxabs(self, vec) -> float:
        L = self.imp.lines
        out = 0.0
        for sid, w in self.ws.items():
            x, xt = self._get(w, vec)
            wp.launch(self.K["linemax"], dim=L[sid].dims,
                      inputs=[x, xt, int(self.coupled), self.wv, self.wnu, self.ng, w["n"], L[sid].la],
                      outputs=[w["lmax"]], device=w["dev"])
            out = max(out, float(w["lmax"].numpy().max()))
        return out

    def combo(self, key: str, coef: np.ndarray, base, alpha: float, out) -> None:
        """out = alpha base + sum_k coef[k] ws[key][k] (key "V" or "Z"; base/out vector refs)."""
        n = len(coef)
        for w in self.ws.values():
            if n:
                w["coef"].assign(np.concatenate([coef, np.zeros(self.m + 1 - n)]))
            b, bt = self._get(w, base)
            o, ot = self._get(w, out)
            wp.launch(self.K["combo"], dim=w["nc"],
                      inputs=[w[key], w[key + "t"], w["coef"], n, int(self.coupled), b, bt, float(alpha), self.ng],
                      outputs=[o, ot], device=w["dev"])

    # ---- operators ---------------------------------------------------------------------------
    def precondition(self, j: int) -> None:
        """Z[j] = M^-1 V[j]: line relaxations from zero with V[j] as right-hand side."""
        imp, cfg = self.imp, self.imp.cfg
        sw = cfg.krylov_sweeps if cfg.krylov_sweeps > 0 else None
        for sid, L in imp.lines.items():
            wp.copy(L.rhs, self.ws[sid]["V"][j])
            L.dq.zero_()
            L.dq2.zero_()
        self.sweeps += imp.relax(sweeps=sw)
        for sid, L in imp.lines.items():
            wp.copy(self.ws[sid]["Z"][j], L.dq)  # with interface ghosts (exchanged by relax)
        if self.coupled:
            t = self.turb
            for sid, w in self.ws.items():
                wp.launch(self.K["to_rhs"], dim=w["nc"], inputs=[w["Vt"][j], self.ng], outputs=[w["rhs_t"]],
                          device=w["dev"])
                t.dn[sid].zero_()
                t.dn2[sid].zero_()
            t.relax(cfg, cfg.smoother == "zebra", rhs={sid: w["rhs_t"] for sid, w in self.ws.items()}, sweeps=sw)
            for sid, w in self.ws.items():
                wp.copy(w["Zt"][j], t.dn[sid])

    def matvec(self, j: int) -> None:
        """w = A Z[j]."""
        imp, s, ng = self.imp, self.s, self.ng
        if not self.jfnk:
            P = imp.I.params
            for sid, d in imp.I.data.items():
                L, m, w = imp.lines[sid], s.metrics[sid], self.ws[sid]
                wp.launch(imp.k_apply, dim=d.sub.nc,
                          inputs=[d.W, d.mut, m.Si, m.Sj, m.Sk, m.vol, ng, wp.vec3i(*d.sub.nc), L.la, L.fcode, P,
                                  imp.jc, L.D, L.Lo, L.Up, w["Z"][j]],
                          outputs=[w["w"]], device=d.sub.device)
            return
        zmax = self.maxabs(("Z", "Zt", j))
        if zmax == 0.0:
            for w in self.ws.values():
                w["w"].zero_()
                w["wt"].zero_()
            return
        eps = imp.cfg.fd_epsilon / zmax
        rs = imp.mp.residual.scalar
        for sid, d in imp.I.data.items():
            w = self.ws[sid]
            wp.launch(self.K["perturb"], dim=d.sub.nc, inputs=[w["U0"], w["Z"][j], rs(eps), ng], outputs=[s.U[sid]],
                      device=d.sub.device)
            if self.coupled:
                wp.launch(self.K["perturb_t"], dim=d.sub.nc, inputs=[w["n0"], w["Zt"][j], rs(eps), ng],
                          outputs=[self.turb.nut[sid]], device=d.sub.device)
        imp.I.rhs(0.0, 0)
        if self.coupled:
            self.turb.residual()
        for sid, d in imp.I.data.items():
            w, L = self.ws[sid], imp.lines[sid]
            wp.launch(self.K["jfnk"], dim=d.sub.nc, inputs=[w["Z"][j], L.dts, d.R, w["R0"], rs(1.0 / eps), ng],
                      outputs=[w["w"]], device=d.sub.device)
            if self.coupled:
                wp.launch(self.K["jfnk_t"], dim=d.sub.nc,
                          inputs=[w["Zt"][j], L.dts, self.turb.R[sid], w["Rt0"], rs(1.0 / eps), ng],
                          outputs=[w["wt"]], device=d.sub.device)

    # ---- solve -------------------------------------------------------------------------------
    def solve(self) -> int:
        """Solve the assembled system: right-hand side R in ``lines[*].rhs`` (and R_nut in
        ``turbulence.R`` when coupled); the correction ends up in ``lines[*].dq`` (and
        ``turbulence.dn``). Returns the number of Krylov iterations."""
        imp, ng, m = self.imp, self.ng, self.m
        self.sweeps = 0
        t = self.turb
        if self.coupled:  # nut scale: max |nut| (a max norm: independent of the partitioning)
            nmax = max(float(np.abs(self.s.interior(a.numpy())).max()) for a in t.nut.values())
            self.wnu = 1.0 / max(nmax, t.nut_inf)
        for sid, w in self.ws.items():
            L = imp.lines[sid]
            wp.copy(w["V"][0], L.rhs)
            wp.launch(self.K["inv"], dim=w["nc"], inputs=[L.dts, ng], outputs=[w["wc"]], device=w["dev"])
            if self.jfnk:
                wp.copy(w["U0"], self.s.U[sid])
                wp.copy(w["R0"], imp.I.data[sid].R)
            if self.coupled:
                wp.copy(w["n0"], t.nut[sid])
                wp.copy(w["Rt0"], t.R[sid])
                wp.launch(self.K["from_rhs"], dim=w["nc"], inputs=[t.R[sid], ng], outputs=[w["Vt"][0]],
                          device=w["dev"])
        V0, Wv, dq = ("V", "Vt", 0), ("w", "wt", None), ("dq", "dn", None)
        beta = self.norm_v(0)
        if beta == 0.0:
            self._zero_correction()
            self.iterations, self.drop = 0, 0.0
            return 0
        self.combo("V", np.zeros(0), V0, 1.0 / beta, V0)
        H = np.zeros((m + 1, m))
        g = np.zeros(m + 1)
        g[0] = beta
        cs, sn = np.zeros(m), np.zeros(m)
        k = 0
        for j in range(m):
            self.precondition(j)
            self.matvec(j)
            h = self.dots(j + 1, Wv)
            self.combo("V", -h, Wv, 1.0, Wv)
            h2 = self.dots(j + 1, Wv)
            self.combo("V", -h2, Wv, 1.0, Wv)
            h = h + h2
            Vn = ("V", "Vt", j + 1)
            self.combo("V", np.zeros(0), Wv, 1.0, Vn)
            hn = self.norm_v(j + 1)
            H[:j + 1, j] = h
            H[j + 1, j] = hn
            if hn > 0.0:
                self.combo("V", np.zeros(0), Vn, 1.0 / hn, Vn)
            for i in range(j):  # apply the previous Givens rotations to the new column
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp
            r = math.hypot(H[j, j], H[j + 1, j])
            cs[j], sn[j] = (1.0, 0.0) if r == 0.0 else (H[j, j] / r, H[j + 1, j] / r)
            H[j, j], H[j + 1, j] = r, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k = j + 1
            if abs(g[k]) <= imp.cfg.krylov_tolerance * beta or hn == 0.0:
                break
        y = np.zeros(k)
        for i in range(k - 1, -1, -1):
            y[i] = (g[i] - H[i, i + 1:k] @ y[i + 1:]) / H[i, i] if H[i, i] != 0.0 else 0.0
        self.stalled = abs(g[k]) > imp.cfg.krylov_stall * beta
        if self.stalled:
            # stagnation (the cycle cannot reach a slow mode, e.g. TMR 2DB 89x41 with m = 10): take the
            # preconditioner step beta z_0, i.e. one defect-correction step, so the outer iteration
            # keeps converging
            y = np.zeros(k)
            y[0] = beta
        if self.jfnk:
            for sid, w in self.ws.items():
                wp.copy(self.s.U[sid], w["U0"])
                if self.coupled:
                    wp.copy(t.nut[sid], w["n0"])
        self._ensure_dq_refs()
        self.combo("Z", y, dq, 0.0, dq)
        self.s.halo.exchange({sid: L.dq for sid, L in imp.lines.items()}, stages=(0,))
        if self.coupled:
            self.s.halo.exchange(t.dn, stages=(0,))
        self.iterations, self.drop = k, abs(g[k]) / beta
        return k

    def _ensure_dq_refs(self) -> None:
        """Point ws[*]["dq"/"dn"] at the current correction arrays (relax may swap buffers)."""
        for sid, w in self.ws.items():
            w["dq"] = self.imp.lines[sid].dq
            w["dn"] = self.turb.dn[sid] if self.coupled else w["wt"]

    def _zero_correction(self) -> None:
        for L in self.imp.lines.values():
            L.dq.zero_()
        if self.coupled:
            for a in self.turb.dn.values():
                a.zero_()
