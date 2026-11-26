"""Order of accuracy of the steady RANS-SA discretization with a manufactured solution.

Curvilinear two-block grid (second block with permuted/flipped index axes), periodic in x,
exact-solution (Dirichlet) ghosts on the other faces, three partitions. The manufactured mean
flow and SA variable (verify/mms.py: sa_solution) are steady; mu_t = rho nut f_v1 enters the
mean-flow viscous terms, the SA source uses an analytic wall distance. Each grid is converged
with the implicit solver (solver.mode: rans, roe_muscl flux, turbulence: sa).

Expected: second order for the mean flow, first order for nut (first-order upwind convection,
as in the NASA TMR verification codes).

    conda run -n pyvista env PYTHONPATH=src python verification/sa_mms.py [--grids 8,16,32]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from helpers import _two_block_case  # noqa: E402

from wmles_wrap.parallel.devices import init_warp  # noqa: E402
from wmles_wrap.verify.mms import sa_solution  # noqa: E402

VARS = ("rho", "u", "v", "w", "p", "nut")


def run_sa_mms(n: int, partitions: int = 3, reynolds: float = 1000.0, max_iterations: int = 400):
    """Returns ({var: (L2, Linf)}, iterations, converged) on a grid with n cells per unit length."""
    tmp = Path(tempfile.mkdtemp())
    dir_bc = [
        {"faces": ["jmin", "jmax", "kmin", "kmax"], "type": "dirichlet", "block": 0},
        {"faces": ["imin", "imax", "jmin", "jmax"], "type": "dirichlet", "block": 1},
    ]
    solver = _two_block_case(
        tmp, partitions, dims=(2 * n + 1, n + 1, n + 1),
        extra={
            "flow": {"mach": 0.5, "reynolds": reynolds, "temperature": 300.0},
            "numerics": {"flux": "roe_muscl"},
            "sgs": {"model": "none"},
            "wall_model": {"type": "none"},
            "boundaries": dir_bc,
            "periodic": [{"translation": [2.0, 0.0, 0.0]}],
            "solver": {"mode": "rans"},
            "rans": {"turbulence": "sa", "max_iterations": max_iterations, "tolerance": 1e-11},
            "output": {"dir": str(tmp / "out")},
            "run": {"log_every": 10**6},
        },
    )
    ms = sa_solution(solver.gas)
    solver.set_primitive_field(lambda xc: ms.primitives(xc, 0.0))
    solver.integrator.exact_fn = ms.primitives
    solver.integrator.source_fn = ms.source
    solver.set_wall_distance(ms.wall_distance)
    T = solver.turbulence
    T.exact_fn = lambda x: ms.nut.value(x, 0.0)
    T.source_fn = lambda x: ms.sa_source(x, 0.0)
    T.load_parent_nut([ms.nut.value(0.125 * sum(x[i:x.shape[0] - 1 + i, j:x.shape[1] - 1 + j, k:x.shape[2] - 1 + k]
                                                  for i in (0, 1) for j in (0, 1) for k in (0, 1)), 0.0)
                       for x in solver.parent_blocks])
    converged = solver.run_steady()
    sq, linf, vol_tot = np.zeros(6), np.zeros(6), 0.0
    for s in solver.subs:
        U = solver.interior(solver.U[s.id].numpy()).astype(np.float64)
        prim = solver.gas.primitives(U)
        xc = solver.interior(solver.metrics[s.id].xc.numpy()).astype(np.float64)
        vol = solver.interior(solver.metrics[s.id].vol.numpy()).astype(np.float64)
        num = np.stack([prim[k] for k in VARS[:5]] + [solver.interior(T.nut[s.id].numpy())], axis=-1)
        ex = np.concatenate([ms.primitives(xc, 0.0), ms.nut.value(xc, 0.0)[..., None]], axis=-1)
        err = num - ex
        sq += (err**2 * vol[..., None]).sum(axis=(0, 1, 2))
        linf = np.maximum(linf, np.abs(err).max(axis=(0, 1, 2)))
        vol_tot += vol.sum()
    l2 = np.sqrt(sq / vol_tot)
    return {k: (l2[i], linf[i]) for i, k in enumerate(VARS)}, solver.step, converged


def orders(grids, res, norm=0):
    out = {}
    for v in VARS:
        e = [r[v][norm] for r in res]
        out[v] = [np.log(e[i - 1] / e[i]) / np.log(grids[i] / grids[i - 1]) for i in range(1, len(grids))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="8,16,32")
    args = ap.parse_args()
    init_warp()
    grids = [int(g) for g in args.grids.split(",")]
    runs = [run_sa_mms(n) for n in grids]
    res = [r[0] for r in runs]
    print("Steady RANS-SA MMS (roe_muscl, first-order nut convection)")
    print(f"{'n':>4} {'iters':>6} " + " ".join(f"{v + ' L2':>11} {'ord':>5}" for v in VARS))
    for i, n in enumerate(grids):
        row = f"{n:4d} {runs[i][1]:6d} "
        for v in VARS:
            e = res[i][v][0]
            o = np.log(res[i - 1][v][0] / e) / np.log(grids[i] / grids[i - 1]) if i else float("nan")
            row += f"{e:11.3e} {o:5.2f} "
        print(row + ("" if runs[i][2] else "  (not converged)"))


if __name__ == "__main__":
    main()
