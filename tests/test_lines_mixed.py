"""Line-preserving partitioning (parallel.keep_lines) and the mixed-precision building blocks of
the implicit solver (5x5 block solve, residual/solve-precision transfers)."""

import numpy as np
import pytest
import warp as wp

from helpers import _two_block_case
from wmles_wrap.grid.partition import line_axes, split_blocks
from wmles_wrap.numerics.mixed import block_kernels, mixed_kernels
from wmles_wrap.precision import get_mixed, get_precision


def _stretched_box(n=(9, 7, 5), axis=1, ratio=1.3, first=1e-4):
    """Unit-ish box, geometric clustering towards index 0 along ``axis``."""
    coords = [np.linspace(0, 1, m) for m in n]
    coords[axis] = np.concatenate([[0.0], first * np.cumsum(ratio ** np.arange(n[axis] - 1))])
    return np.stack(np.meshgrid(*coords, indexing="ij"), -1)


# ---- line axes and partitioning -----------------------------------------------------------------

def test_line_axes():
    a = _stretched_box(axis=1)  # clustered in j, no walls -> j from the geometry
    b = _stretched_box(axis=2)  # clustered in k, but its wall is on imin -> i
    c = _stretched_box(n=(2, 9, 7), axis=0, first=1e-6)  # one cell thick in i -> never i; j finer than k
    d = _stretched_box(axis=2)  # walls on jmin and kmin -> geometric choice among j, k
    walls = {(1, 0), (3, 2), (3, 4)}
    assert line_axes([a, b, c, d], walls) == [1, 0, 1, 2]


@pytest.mark.parametrize("nparts", [2, 5, 9, 16])
def test_split_keeps_lines(nparts):
    rng = np.random.default_rng(nparts)
    blocks = [np.zeros(tuple(int(v) for v in rng.integers(9, 40, 3)) + (3,)) for _ in range(3)]
    keep = [0, 2, 1]
    subs = split_blocks(blocks, nparts, min_cells=4, keep_axes=keep)
    assert len(subs) >= min(nparts, 3)
    for s in subs:
        ax = keep[s.parent]
        assert s.lo[ax] == 0 and s.hi[ax] == s.parent_nn[ax] - 1
    # the cells still tile every parent block exactly
    for b, x in enumerate(blocks):
        assert sum(s.ncells for s in subs if s.parent == b) == np.prod([n - 1 for n in x.shape[:3]])
    # without keep_axes the partitioner is unchanged
    plain = split_blocks(blocks, nparts, min_cells=4)
    plain2 = split_blocks(blocks, nparts, min_cells=4, keep_axes=None)
    assert [(s.parent, s.lo, s.hi) for s in plain] == [(s.parent, s.lo, s.hi) for s in plain2]


def test_solver_keep_lines_partition_independent(tmp_path):
    """With keep_lines the line axis of every parent stays whole in every sub-block, and a short
    run is bitwise independent of the partitioning."""
    states = []
    for n in (1, 4):
        s = _two_block_case(tmp_path, n, extra={"parallel": {"keep_lines": True},
                                                 "run": {"steps": 3}, "output": {"dir": str(tmp_path / f"o{n}")}})
        assert s.line_axes == [1, 0]  # block 0 walls on j, block 1 (reoriented) walls on i
        for sb in s.subs:
            ax = s.line_axes[sb.parent]
            assert sb.lo[ax] == 0 and sb.hi[ax] == sb.parent_nn[ax] - 1
        assert "line axes" in s.summary()
        s.run()
        states.append(s.parent_states())
    assert len(states) == 2 and all(np.array_equal(a, b) for a, b in zip(*states))


# ---- mixed precision ----------------------------------------------------------------------------

@pytest.mark.parametrize("precision, tol", [("float64", 1e-12), ("float32", 2e-4)])
def test_block_solve(precision, tol):
    prec = get_precision(precision)
    _, k = block_kernels(prec)
    rng = np.random.default_rng(4)
    n = 200
    A = rng.standard_normal((n, 5, 5)) + 3.0 * np.eye(5)
    A[0] = [[0, 1, 2, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0], [2, 0, 1, 0, 0], [0, 1, 0, 0, 3]]  # needs pivoting
    b = rng.standard_normal((n, 5))
    want = np.linalg.solve(A, b[..., None])[..., 0]
    x = wp.zeros(n, dtype=prec.vec5)
    wp.launch(k, dim=n, inputs=[wp.array(A.astype(prec.np_dtype), dtype=prec.mat55),
                                wp.array(b.astype(prec.np_dtype), dtype=prec.vec5)], outputs=[x])
    got = x.numpy()
    assert np.abs(got - want).max() <= tol * np.abs(want).max()


@pytest.mark.parametrize("residual", ["float64", "float32"])
def test_mixed_transfers(residual):
    mp = get_mixed(residual, "float32")
    k_rhs, k_correct = mixed_kernels(mp)
    ng, nc = 2, (3, 4, 2)
    shape = tuple(n + 2 * ng for n in nc)
    rng = np.random.default_rng(5)
    R = rng.standard_normal(shape + (5,))
    Q = rng.standard_normal(shape + (5,))
    rhs = wp.full(shape, mp.solve.vec5(7.0, 7.0, 7.0, 7.0, 7.0), dtype=mp.solve.vec5)
    wp.launch(k_rhs, dim=nc, inputs=[wp.array(R.astype(mp.residual.np_dtype), dtype=mp.residual.vec5), ng],
              outputs=[rhs])
    r = rhs.numpy()
    inner = (slice(ng, -ng),) * 3
    assert r.dtype == np.float32
    assert np.array_equal(r[inner], (-R[inner].astype(mp.residual.np_dtype)).astype(np.float32))
    ghost = np.ones(shape, dtype=bool)
    ghost[inner] = False
    assert np.all(r[ghost] == 7.0)  # ghosts untouched
    Qd = wp.array(Q.astype(mp.residual.np_dtype), dtype=mp.residual.vec5)
    wp.launch(k_correct, dim=nc, inputs=[rhs, ng], outputs=[Qd])
    q = Qd.numpy()
    want = Q.astype(mp.residual.np_dtype)
    want[inner] += r[inner].astype(mp.residual.np_dtype)
    assert np.array_equal(q, want)
