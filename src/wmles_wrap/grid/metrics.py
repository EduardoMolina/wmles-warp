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

All geometry is differenced in float64 from float64 node coordinates and then stored in the
run precision. In float32, differencing coordinates of magnitude L loses digits on cells of
size h ~ L * 1e-7 (e.g. a 1e-6 c first cell 30 c from the origin); stored values keep the
float32 relative precision of the float64 result. The GCL diagnostic is evaluated on the
stored face vectors, i.e. what the solver sees.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..precision import Precision, get_precision, kernel_factory


@kernel_factory
def _metric_kernels(prec: Precision):
    """Differencing kernels; always instantiated with float64 (see compute_metrics)."""
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
    ):
        p, q, r = wp.tid()
        v = (
            wp.dot(Si[p + 1, q, r], ci[p + 1, q, r]) - wp.dot(Si[p, q, r], ci[p, q, r])
            + wp.dot(Sj[p, q + 1, r], cj[p, q + 1, r]) - wp.dot(Sj[p, q, r], cj[p, q, r])
            + wp.dot(Sk[p, q, r + 1], ck[p, q, r + 1]) - wp.dot(Sk[p, q, r], ck[p, q, r])
        )
        vol[p, q, r] = third * v
        xc[p, q, r] = eighth * (
            X[p, q, r] + X[p + 1, q, r] + X[p, q + 1, r] + X[p + 1, q + 1, r]
            + X[p, q, r + 1] + X[p + 1, q, r + 1] + X[p, q + 1, r + 1] + X[p + 1, q + 1, r + 1]
        )

    return face_i, face_j, face_k, cells


@kernel_factory
def _store_kernels(prec: Precision):
    """Kernels in the run precision: float64 -> run-precision copies and the GCL diagnostic."""
    vec3 = prec.vec3
    scalar = prec.scalar

    @wp.kernel(module="unique")
    def cast_vec3(src: wp.array3d(dtype=wp.vec3d), dst: wp.array3d(dtype=vec3)):
        p, q, r = wp.tid()
        v = src[p, q, r]
        dst[p, q, r] = vec3(scalar(v[0]), scalar(v[1]), scalar(v[2]))

    @wp.kernel(module="unique")
    def cast_scalar(src: wp.array3d(dtype=wp.float64), dst: wp.array3d(dtype=scalar)):
        p, q, r = wp.tid()
        dst[p, q, r] = scalar(src[p, q, r])

    @wp.kernel(module="unique")
    def gcl_k(
        Si: wp.array3d(dtype=vec3),
        Sj: wp.array3d(dtype=vec3),
        Sk: wp.array3d(dtype=vec3),
        gcl: wp.array3d(dtype=scalar),
    ):
        p, q, r = wp.tid()
        s = Si[p + 1, q, r] - Si[p, q, r] + Sj[p, q + 1, r] - Sj[p, q, r] + Sk[p, q, r + 1] - Sk[p, q, r]
        ref = (
            wp.length(Si[p, q, r]) + wp.length(Si[p + 1, q, r]) + wp.length(Sj[p, q, r])
            + wp.length(Sj[p, q + 1, r]) + wp.length(Sk[p, q, r]) + wp.length(Sk[p, q, r + 1])
        )
        gcl[p, q, r] = wp.length(s) / ref

    return cast_vec3, cast_scalar, gcl_k


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
    """Compute metrics for padded node coordinates ``xp`` of shape (Ni+1, Nj+1, Nk+1, 3).
    Differencing is done in float64; the arrays are returned in ``prec``."""
    f64 = get_precision("float64")
    face_i, face_j, face_k, cells = _metric_kernels(f64)
    cast_vec3, cast_scalar, gcl_k = _store_kernels(prec)
    nn = xp.shape[:3]
    nc = tuple(n - 1 for n in nn)
    X = wp.array(np.asarray(xp, dtype=np.float64), dtype=wp.vec3d, device=device)
    z3 = lambda shape: wp.zeros(shape, dtype=wp.vec3d, device=device)  # noqa: E731
    Si, ci = z3((nn[0], nc[1], nc[2])), z3((nn[0], nc[1], nc[2]))
    Sj, cj = z3((nc[0], nn[1], nc[2])), z3((nc[0], nn[1], nc[2]))
    Sk, ck = z3((nc[0], nc[1], nn[2])), z3((nc[0], nc[1], nn[2]))
    wp.launch(face_i, dim=Si.shape, inputs=[X], outputs=[Si, ci], device=device)
    wp.launch(face_j, dim=Sj.shape, inputs=[X], outputs=[Sj, cj], device=device)
    wp.launch(face_k, dim=Sk.shape, inputs=[X], outputs=[Sk, ck], device=device)
    vol = wp.zeros(nc, dtype=wp.float64, device=device)
    xc = z3(nc)
    wp.launch(cells, dim=nc, inputs=[X, Si, Sj, Sk, ci, cj, ck], outputs=[vol, xc], device=device)
    del X, ci, cj, ck
    if prec.name != "float64":

        def store(a):
            out = wp.zeros(a.shape, dtype=prec.vec3 if a.dtype == wp.vec3d else prec.scalar, device=device)
            wp.launch(cast_vec3 if a.dtype == wp.vec3d else cast_scalar, dim=a.shape, inputs=[a], outputs=[out],
                      device=device)
            return out

        Si, Sj, Sk, vol, xc = (store(a) for a in (Si, Sj, Sk, vol, xc))
    gcl = wp.zeros(nc, dtype=prec.scalar, device=device)
    wp.launch(gcl_k, dim=nc, inputs=[Si, Sj, Sk], outputs=[gcl], device=device)
    return Metrics(Si, Sj, Sk, vol, xc, gcl)
