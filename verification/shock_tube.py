"""1D shock tubes (Sod, Lax, strong 1000:1 pressure ratio) vs the exact Riemann solution.

    conda run -n pyvista env PYTHONPATH=src python verification/shock_tube.py [--case sod] [--n 200] [--plot]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from wmles_wrap.config import config_from_dict
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.parallel.devices import init_warp
from wmles_wrap.solver import Solver
from wmles_wrap.verify.riemann import exact_riemann

CASES = {  # (rho, u, p) left, right, t_end
    "sod": ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2),
    "lax": ((0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 0.14),
    "strong": ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.012),
}


def run_shock_tube(case="sod", n=200, partitions=2, dissipation="roe", sensor="ducros_pressure", cfl=0.5,
                   precision="float64"):
    left, right, t_end = CASES[case]
    tmp = Path(tempfile.mkdtemp())
    ny = 2
    x = np.stack(np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, ny / n, ny + 1),
                             np.linspace(0, ny / n, ny + 1), indexing="ij"), axis=-1)
    write_plot3d(tmp / "tube.p3d", [x])
    cfg = config_from_dict(
        {
            "grid": {"file": "tube.p3d"},
            # Euler equations are scale invariant: states are given directly in solver units
            "flow": {"mach": 1.0, "reynolds": 1.0, "temperature": 300.0},
            "numerics": {"viscous": False, "dissipation": dissipation, "sensor": sensor, "cfl": cfl,
                         "precision": precision},
            "boundaries": [{"faces": ["imin", "imax"], "type": "extrapolate"}],
            "periodic": [{"translation": [0, ny / n, 0]}, {"translation": [0, 0, ny / n]}],
            "parallel": {"devices": ["cpu"], "partitions": partitions},
        },
        case_dir=tmp,
    )
    solver = Solver(cfg)

    def init(xc):
        W = np.zeros(xc.shape[:-1] + (5,))
        L = xc[..., 0] < 0.5
        for k, idx in ((0, 0), (1, 1), (4, 2)):
            W[..., k] = np.where(L, left[idx], right[idx])
        return W

    solver.set_primitive_field(init)
    solver.advance(t_end=t_end)
    U = solver.parent_states()[0][:, 0, 0].astype(np.float64)
    prim = solver.gas.primitives(U)
    xc = 0.5 * (x[1:, 0, 0, 0] + x[:-1, 0, 0, 0])
    rho_e, u_e, p_e = exact_riemann(left, right, solver.gas.gamma, xc, solver.time)
    errs = {
        "rho": np.mean(np.abs(prim["rho"] - rho_e)) / np.mean(np.abs(rho_e)),
        "u": np.mean(np.abs(prim["u"] - u_e)) / max(np.mean(np.abs(u_e)), 1e-12),
        "p": np.mean(np.abs(prim["p"] - p_e)) / np.mean(np.abs(p_e)),
    }
    return xc, prim, (rho_e, u_e, p_e), errs, solver.step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="sod", choices=sorted(CASES))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dissipation", default="roe")
    ap.add_argument("--sensor", default="ducros_pressure")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    init_warp()
    xc, prim, ex, errs, steps = run_shock_tube(args.case, args.n, dissipation=args.dissipation, sensor=args.sensor)
    print(f"{args.case} n={args.n} steps={steps}  relative L1 errors: "
          + "  ".join(f"{k} {v:.3e}" for k, v in errs.items())
          + f"  min rho {prim['rho'].min():.3e} min p {prim['p'].min():.3e}")
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        for a, k, e in zip(ax, ("rho", "u", "p"), ex):
            a.plot(xc, e, "k-", lw=1, label="exact")
            a.plot(xc, prim[k], "o", ms=2.5, label="wmles_wrap")
            a.set_title(k)
        ax[0].legend()
        out = Path(__file__).parent / "results" / f"shock_tube_{args.case}_{args.n}.png"
        out.parent.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
