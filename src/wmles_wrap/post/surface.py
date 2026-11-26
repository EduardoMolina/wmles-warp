"""Surface quantities on resolved (no-slip) walls: pressure, shear stress, C_p, C_f, forces.

Host-side float64 evaluation from the parent-block states. The wall-normal derivative of the
velocity uses the first two cell centroids (``order=2``, one-sided quadratic through u_wall at the
wall: du/dn = (du_1 d_2^2 - du_2 d_1^2) / (d_1 d_2 (d_2 - d_1)), du_k = u_k - u_wall) or the first
cell only (``order=1``, du_1 / d_1, the value the solver's wall flux uses with mirrored ghosts).
d_k is the normal distance of cell centroid k from the wall face. Wall pressure and (adiabatic)
wall temperature are extrapolated with zero normal gradient to the same order.

Nondimensional: free-stream rho = |u| = 1, so q_inf = 1/2, C_f = tau / q_inf and
C_p = (p_w - p_inf) / q_inf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid.wall_distance import wall_faces


@dataclass
class WallData:
    xf: np.ndarray  # (n1, n2, 3) face centroids
    S: np.ndarray  # (n1, n2, 3) area vectors, pointing into the fluid
    p: np.ndarray  # (n1, n2) wall pressure
    T: np.ndarray  # (n1, n2) wall temperature
    tau: np.ndarray  # (n1, n2, 3) shear stress exerted by the fluid on the wall
    cf: np.ndarray  # (n1, n2) |tau| / q_inf, signed by tau . t_ref when a reference direction is given
    cp: np.ndarray  # (n1, n2)

    def force(self, p_inf: float) -> np.ndarray:
        """Total force (3,) of the fluid on this face: shear plus pressure relative to p_inf."""
        A = np.linalg.norm(self.S, axis=-1)[..., None]
        return (self.tau * A - (self.p - p_inf)[..., None] * self.S).reshape(-1, 3).sum(axis=0)


def _centroids(x: np.ndarray, ax: int, layers: tuple[int, int]) -> list[np.ndarray]:
    """Cell centroids (8-node averages) of two cell layers normal to axis ax."""
    out = []
    for c in layers:
        xs = np.take(x, [c, c + 1], axis=ax)
        cc = 0.125 * sum(xs[i:xs.shape[0] - 1 + i, j:xs.shape[1] - 1 + j, k:xs.shape[2] - 1 + k]
                         for i in (0, 1) for j in (0, 1) for k in (0, 1))
        out.append(np.take(cc, 0, axis=ax))
    return out


def wall_data(solver, order: int = 2, direction=None) -> dict[tuple[int, int], WallData]:
    """(parent block, face) -> WallData on every wall face. ``direction`` (3,) signs C_f by the
    shear component along it (default: the free-stream direction)."""
    gas, cfg = solver.gas, solver.cfg
    rho, u, v, w, T = gas.freestream(cfg.flow.alpha, cfg.flow.beta)
    p_inf, q_inf = rho * gas.R * T, 0.5 * rho * (u * u + v * v + w * w)
    tref = np.asarray(direction if direction is not None else (u, v, w), dtype=np.float64)
    tref = tref / np.linalg.norm(tref)
    states = solver.parent_states()
    bcs = {(s.parent, f): bc for s in solver.subs for f, bc in s.face_bc.items() if bc is not None}
    out = {}
    for b, f in wall_faces(solver.subs):
        x, U = solver.parent_blocks[b], states[b]
        ax, side = f // 2, f % 2
        nc = U.shape[ax]
        if order == 2 and nc < 2:
            raise ValueError(f"order 2 wall data needs 2 cells normal to block {b} face {f}")
        xf, S = solver._wall_face_geometry(b, f)
        n = S / np.linalg.norm(S, axis=-1, keepdims=True)
        lay = (0, 1) if side == 0 else (nc - 1, nc - 2)
        c1, c2 = _centroids(x, ax, lay)
        d1 = np.sum((c1 - xf) * n, axis=-1)
        d2 = np.sum((c2 - xf) * n, axis=-1)
        W1 = gas.primitives(np.take(U, lay[0], axis=ax))
        W2 = gas.primitives(np.take(U, lay[1], axis=ax))
        bc = bcs[(b, f)]
        uw = np.asarray(bc.params.get("u_wall", (0.0, 0.0, 0.0)), dtype=np.float64)
        du1 = np.stack([W1["u"], W1["v"], W1["w"]], -1) - uw
        du2 = np.stack([W2["u"], W2["v"], W2["w"]], -1) - uw
        if order == 2:
            e1, e2 = (d2 * d2)[..., None], (d1 * d1)[..., None]
            dudn = (du1 * e1 - du2 * e2) / (d1 * d2 * (d2 - d1))[..., None]
            zg = lambda a1, a2: (a1 * d2 * d2 - a2 * d1 * d1) / (d2 * d2 - d1 * d1)  # noqa: E731
        else:
            dudn = du1 / d1[..., None]
            zg = lambda a1, a2: a1  # noqa: E731
        pw = zg(W1["p"], W2["p"])
        Tw = np.full_like(pw, float(bc.T_wall)) if bc.thermal == "isothermal" else zg(W1["T"], W2["T"])
        dudn_t = dudn - np.sum(dudn * n, axis=-1, keepdims=True) * n
        tau = gas.mu(Tw)[..., None] * dudn_t
        sgn = np.where(np.sum(tau * tref, axis=-1) < 0, -1.0, 1.0)
        out[(b, f)] = WallData(xf, S, pw, Tw, tau, sgn * np.linalg.norm(tau, axis=-1) / q_inf,
                               (pw - p_inf) / q_inf)
    return out
