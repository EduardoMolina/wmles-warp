"""Finite-volume geometric metrics on hexahedral cells (Warp kernels).

All arrays are padded (ng ghost layers). For a block with padded cell extents (Ni, Nj, Nk)
(padded node extents Ni+1, ...):
    Si : (Ni+1, Nj, Nk)  i-face area vectors, face p separates cells p-1 and p, normal +i
    Sj : (Ni, Nj+1, Nk)  j-face area vectors, normal +j
    Sk : (Ni, Nj, Nk+1)  k-face area vectors, normal +k
    vol: (Ni, Nj, Nk)    cell volumes
    xc : (Ni, Nj, Nk)    cell centroids (mean of the 8 vertices)

Face area vectors use the diagonal cross-product formula S = 1/2 (d1 x d2), which is exact
for planar faces and makes the sum of outward face vectors of every cell vanish identically
(discrete geometric conservation law), independent of face warping. Volumes use the
divergence theorem V = 1/3 sum_f S_f . x_f with x_f the face vertex mean.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..precision import Precision, kernel_factory


@kernel_factory
def _metric_kernels(prec: Precision):
    vec3 = prec.vec3
    scalar = prec.scalar
    half = scalar(0.5)
    quarter = scalar(0.25)
    third = scalar(1.0 / 3.0)
    eighth = scalar(0.125)

    @wp.kernel(module="unique")
    def face_i(X: wp.array3d(dtype=vec3), Si: wp.array3d(dtype=vec3), ci: wp.array3d(dtype=vec3)):
        p, q, r = wp.tid()
        a = X[p, q, r]
        b = X[p, q + 1, r]
        c = X[p, q + 1, r + 1]
        d = X[p, q, r + 1]
        Si[p, q, r] = half * wp.cross(c - a, d - b)
        ci[p, q, r] = quarter * (a + b + c + d)

    @wp.kernel(module="unique")
    def face_j(X: wp.array3d(dtype=vec3), Sj: wp.array3d(dtype=vec3), cj: wp.array3d(dtype=vec3)):
        p, q, r = wp.tid()
        a = X[p, q, r]
        b = X[p, q, r + 1]
        c = X[p + 1, q, r + 1]
        d = X[p + 1, q, r]
        Sj[p, q, r] = half * wp.cross(c - a, d - b)
        cj[p, q, r] = quarter * (a + b + c + d)

    @wp.kernel(module="unique")
    def face_k(X: wp.array3d(dtype=vec3), Sk: wp.array3d(dtype=vec3), ck: wp.array3d(dtype=vec3)):
        p, q, r = wp.tid()
        a = X[p, q, r]
        b = X[p + 1, q, r]
        c = X[p + 1, q + 1, r]
        d = X[p, q + 1, r]
        Sk[p, q, r] = half * wp.cross(c - a, d - b)
        ck[p, q, r] = quarter * (a + b + c + d)

    @wp.kernel(module="unique")
    def cells(
        X: wp.array3d(dtype=vec3),
        Si: wp.array3d(dtype=vec3),
        Sj: wp.array3d(dtype=vec3),
        Sk: wp.array3d(dtype=vec3),
        ci: wp.array3d(dtype=vec3),
        cj: wp.array3d(dtype=vec3),
        ck: wp.array3d(dtype=vec3),
        vol: wp.array3d(dtype=scalar),
        xc: wp.array3d(dtype=vec3),
        gcl: wp.array3d(dtype=scalar),
    ):
        p, q, r = wp.tid()
        v = (
            wp.dot(Si[p + 1, q, r], ci[p + 1, q, r]) - wp.dot(Si[p, q, r], ci[p, q, r])
            + wp.dot(Sj[p, q + 1, r], cj[p, q + 1, r]) - wp.dot(Sj[p, q, r], cj[p, q, r])
            + wp.dot(Sk[p, q, r + 1], ck[p, q, r + 1]) - wp.dot(Sk[p, q, r], ck[p, q, r])
        )
        vol[p, q, r] = third * v
        s = Si[p + 1, q, r] - Si[p, q, r] + Sj[p, q + 1, r] - Sj[p, q, r] + Sk[p, q, r + 1] - Sk[p, q, r]
        ref = (
            wp.length(Si[p, q, r]) + wp.length(Si[p + 1, q, r]) + wp.length(Sj[p, q, r])
            + wp.length(Sj[p, q + 1, r]) + wp.length(Sk[p, q, r]) + wp.length(Sk[p, q, r + 1])
        )
        gcl[p, q, r] = wp.length(s) / ref
        xc[p, q, r] = eighth * (
            X[p, q, r] + X[p + 1, q, r] + X[p, q + 1, r] + X[p + 1, q + 1, r]
            + X[p, q, r + 1] + X[p + 1, q, r + 1] + X[p, q + 1, r + 1] + X[p + 1, q + 1, r + 1]
        )

    return face_i, face_j, face_k, cells


@dataclass
class Metrics:
    Si: wp.array
    Sj: wp.array
    Sk: wp.array
    vol: wp.array
    xc: wp.array
    gcl: wp.array  # |sum of outward face vectors| / sum |S_f|, per cell (diagnostic)

    def gcl_max(self) -> float:
        g = self.gcl.numpy()
        return float(g.max()) if g.size else 0.0

    def min_volume(self, ng: int, interior_only: bool = True) -> float:
        v = self.vol.numpy()
        if interior_only:
            v = v[ng:-ng, ng:-ng, ng:-ng]
        return float(v.min())


def compute_metrics(xp: np.ndarray, prec: Precision, device: str) -> Metrics:
    """Compute metrics for padded node coordinates ``xp`` of shape (Ni+1, Nj+1, Nk+1, 3)."""
    face_i, face_j, face_k, cells = _metric_kernels(prec)
    nn = xp.shape[:3]
    nc = tuple(n - 1 for n in nn)
    X = wp.array(xp.astype(prec.np_dtype), dtype=prec.vec3, device=device)
    z3 = lambda shape: wp.zeros(shape, dtype=prec.vec3, device=device)  # noqa: E731
    Si, ci = z3((nn[0], nc[1], nc[2])), z3((nn[0], nc[1], nc[2]))
    Sj, cj = z3((nc[0], nn[1], nc[2])), z3((nc[0], nn[1], nc[2]))
    Sk, ck = z3((nc[0], nc[1], nn[2])), z3((nc[0], nc[1], nn[2]))
    wp.launch(face_i, dim=Si.shape, inputs=[X], outputs=[Si, ci], device=device)
    wp.launch(face_j, dim=Sj.shape, inputs=[X], outputs=[Sj, cj], device=device)
    wp.launch(face_k, dim=Sk.shape, inputs=[X], outputs=[Sk, ck], device=device)
    vol = wp.zeros(nc, dtype=prec.scalar, device=device)
    gcl = wp.zeros(nc, dtype=prec.scalar, device=device)
    xc = z3(nc)
    wp.launch(cells, dim=nc, inputs=[X, Si, Sj, Sk, ci, cj, ck], outputs=[vol, xc, gcl], device=device)
    return Metrics(Si, Sj, Sk, vol, xc, gcl)
