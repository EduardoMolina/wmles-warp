"""Mixed-precision building blocks of the implicit (defect-correction) solver.

The outer loop evaluates the residual R(Q) in the residual precision (float64 by default); the
inner loop relaxes the linear system in the solve precision (float32) on 5x5 blocks:

    rhs = -R(Q)            (residual -> solve precision, ``k_rhs``)
    relax  M dQ = rhs      (solve precision; block solves with ``block_solve``)
    Q <- Q + dQ            (solve -> residual precision, ``k_correct``)

The inner precision only changes the convergence rate: the converged state satisfies R(Q) = 0
in the residual precision. Warp's ``wp.inverse`` stops at 4x4, so 5x5 blocks are solved by
Gaussian elimination with partial pivoting (``block_solve``).
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

import warp as wp

from ..precision import MixedPrecision, Precision, kernel_factory


@kernel_factory
def block_kernels(prec: Precision):
    """``block_solve(A, b)`` (5x5 dense, partial pivoting) and a test kernel ``k_block_solve``."""
    vec5, mat55, scalar = prec.vec5, prec.mat55, prec.scalar
    one = scalar(1.0)

    @wp.func
    def block_solve(A: mat55, b: vec5):
        M = mat55(A)
        x = vec5(b)
        for k in range(5):
            p = int(k)
            amax = wp.abs(M[k, k])
            for i in range(k + 1, 5):
                if wp.abs(M[i, k]) > amax:
                    amax = wp.abs(M[i, k])
                    p = i
            if p != k:
                for j in range(5):
                    t = M[k, j]
                    M[k, j] = M[p, j]
                    M[p, j] = t
                t = x[k]
                x[k] = x[p]
                x[p] = t
            inv = one / M[k, k]
            for i in range(k + 1, 5):
                f = M[i, k] * inv
                for j in range(k, 5):
                    M[i, j] = M[i, j] - f * M[k, j]
                x[i] = x[i] - f * x[k]
        for kk in range(5):
            k = 4 - kk
            s = x[k]
            for j in range(k + 1, 5):
                s = s - M[k, j] * x[j]
            x[k] = s / M[k, k]
        return x

    @wp.kernel(module="unique")
    def k_block_solve(A: wp.array(dtype=mat55), b: wp.array(dtype=vec5), x: wp.array(dtype=vec5)):
        i = wp.tid()
        x[i] = block_solve(A[i], b[i])

    return block_solve, k_block_solve


@kernel_factory
def mixed_kernels(mp: MixedPrecision):
    """Transfers between the residual and the solve precision on padded cell arrays."""
    rv, sv = mp.residual.vec5, mp.solve.vec5
    rs, ss = mp.residual.scalar, mp.solve.scalar

    @wp.kernel(module="unique")
    def k_rhs(R: wp.array3d(dtype=rv), ng: int, rhs: wp.array3d(dtype=sv)):
        """rhs = -R on the interior cells (solve precision)."""
        i, j, k = wp.tid()
        r = R[i + ng, j + ng, k + ng]
        rhs[i + ng, j + ng, k + ng] = sv(-ss(r[0]), -ss(r[1]), -ss(r[2]), -ss(r[3]), -ss(r[4]))

    @wp.kernel(module="unique")
    def k_correct(dQ: wp.array3d(dtype=sv), ng: int, Q: wp.array3d(dtype=rv)):
        """Q += dQ on the interior cells (residual precision)."""
        i, j, k = wp.tid()
        d = dQ[i + ng, j + ng, k + ng]
        Q[i + ng, j + ng, k + ng] = Q[i + ng, j + ng, k + ng] + rv(rs(d[0]), rs(d[1]), rs(d[2]), rs(d[3]), rs(d[4]))

    return k_rhs, k_correct
