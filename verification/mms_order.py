"""Order-of-accuracy study with the method of manufactured solutions (MMS).

Curvilinear two-block grid (second block with permuted/flipped index axes), periodic in x,
exact-solution (Dirichlet) ghost cells on the y/z faces, decomposed into several partitions.
Unsteady manufactured solution advanced to t_end with dt proportional to h.

    conda run -n pyvista env PYTHONPATH=src python verification/mms_order.py [--euler] [--dissipation roe]
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
from wmles_wrap.verify.mms import default_solution  # noqa: E402

VARS = ("rho", "u", "v", "w", "p")


def run_mms(n: int, viscous: bool = True, dissipation: str = "none", partitions: int = 3,
            t_end: float = 0.1, precision: str = "float64") -> dict[str, tuple[float, float]]:
    """Returns {var: (L2 error, Linf error)} at t_end on a grid with n cells per unit length."""
    tmp = Path(tempfile.mkdtemp())
    dir_bc = [
        {"faces": ["jmin", "jmax", "kmin", "kmax"], "type": "dirichlet", "block": 0},
        {"faces": ["imin", "imax", "jmin", "jmax"], "type": "dirichlet", "block": 1},
    ]
    solver = _two_block_case(
        tmp, partitions, dims=(2 * n + 1, n + 1, n + 1),
        extra={
            "flow": {"mach": 0.5, "reynolds": 50.0, "temperature": 300.0},
            "numerics": {"viscous": viscous, "dissipation": dissipation, "precision": precision},
            "boundaries": dir_bc,
            "periodic": [{"translation": [2.0, 0.0, 0.0]}],
            "run": {"dt": 0.05 / n},
        },
    )
    ms = default_solution(solver.gas, viscous=viscous)
    solver.set_primitive_field(lambda xc: ms.primitives(xc, 0.0))
    solver.integrator.exact_fn = ms.primitives
    solver.integrator.source_fn = ms.source
    solver.advance(t_end=t_end)
    sq = np.zeros(5)
    linf = np.zeros(5)
    vol_tot = 0.0
    for s in solver.subs:
        U = solver.interior(solver.U[s.id].numpy()).astype(np.float64)
        prim = solver.gas.primitives(U)
        num = np.stack([prim[k] for k in VARS], axis=-1)
        xc = solver.interior(solver.metrics[s.id].xc.numpy()).astype(np.float64)
        vol = solver.interior(solver.metrics[s.id].vol.numpy()).astype(np.float64)
        err = num - ms.primitives(xc, solver.time)
        sq += (err**2 * vol[..., None]).sum(axis=(0, 1, 2))
        linf = np.maximum(linf, np.abs(err).max(axis=(0, 1, 2)))
        vol_tot += vol.sum()
    l2 = np.sqrt(sq / vol_tot)
    return {k: (l2[i], linf[i]) for i, k in enumerate(VARS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euler", action="store_true")
    ap.add_argument("--dissipation", default="none")
    ap.add_argument("--grids", default="8,16,32")
    ap.add_argument("--precision", default="float64")
    args = ap.parse_args()
    init_warp()
    grids = [int(g) for g in args.grids.split(",")]
    res = [run_mms(n, viscous=not args.euler, dissipation=args.dissipation, precision=args.precision) for n in grids]
    print(f"MMS {'Euler' if args.euler else 'Navier-Stokes'}, dissipation={args.dissipation}")
    print(f"{'n':>4} " + " ".join(f"{v + ' L2':>11} {'ord':>5}" for v in VARS))
    for i, n in enumerate(grids):
        row = f"{n:4d} "
        for v in VARS:
            e = res[i][v][0]
            o = np.log(res[i - 1][v][0] / e) / np.log(grids[i] / grids[i - 1]) if i else float("nan")
            row += f"{e:11.3e} {o:5.2f} "
        print(row)
    print("Linf:")
    for i, n in enumerate(grids):
        row = f"{n:4d} "
        for v in VARS:
            e = res[i][v][1]
            o = np.log(res[i - 1][v][1] / e) / np.log(grids[i] / grids[i - 1]) if i else float("nan")
            row += f"{e:11.3e} {o:5.2f} "
        print(row)


if __name__ == "__main__":
    main()
