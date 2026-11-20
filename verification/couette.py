"""Compressible plane Couette flow with viscous heating (steady, laminar), exact solution:

    u = U y / h,   T = T_w + Pr U^2 / (2 c_p) (y/h) (1 - y/h),   p = const

for constant viscosity and isothermal walls at T_w (bottom at rest, top moving at U).
Exercises viscous stresses, heat conduction, isothermal and moving no-slip walls.

    conda run -n pyvista env PYTHONPATH=src python verification/couette.py [--grids 8,16,32]
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


def run_couette(ny=16, mach=0.8, reynolds=10.0, t_end=60.0, partitions=1, stretch=0.0, precision="float64"):
    tmp = Path(tempfile.mkdtemp())
    eta = np.linspace(0, 1, ny + 1)
    y = eta + stretch * np.sin(np.pi * eta) / np.pi * 0.0 if stretch == 0 else 0.5 * (1 + np.tanh(stretch * (eta - 0.5)) / np.tanh(0.5 * stretch))
    x = np.stack(np.meshgrid(np.linspace(0, 1, 5), y, np.linspace(0, 1, 5), indexing="ij"), axis=-1)
    write_plot3d(tmp / "couette.p3d", [x])
    cfg = config_from_dict(
        {
            "grid": {"file": "couette.p3d"},
            "flow": {"mach": mach, "reynolds": reynolds, "temperature": 300.0},
            "gas": {"viscosity": "constant"},
            "numerics": {"precision": precision, "cfl": 0.8},
            "sgs": {"model": "none"},
            "boundaries": [
                {"faces": ["jmin"], "type": "wall", "thermal": "isothermal", "T_wall": 1.0},
                {"faces": ["jmax"], "type": "wall", "thermal": "isothermal", "T_wall": 1.0,
                 "params": {"u_wall": [1.0, 0.0, 0.0]}},
            ],
            "periodic": [{"translation": [1, 0, 0]}, {"translation": [0, 0, 1]}],
            "parallel": {"devices": ["cpu"], "partitions": partitions},
        },
        case_dir=tmp,
    )
    solver = Solver(cfg)
    gas = solver.gas

    def init(xc):
        W = np.zeros(xc.shape[:-1] + (5,))
        W[..., 0] = 1.0
        W[..., 1] = xc[..., 1]
        W[..., 4] = gas.R
        return W

    solver.set_primitive_field(init)
    solver.advance(t_end=t_end)
    U = solver.parent_states()[0][2, :, 2].astype(np.float64)
    prim = gas.primitives(U)
    yc = solver.interior(solver.metrics[solver.subs[0].id].xc.numpy())[2, :, 2, 1] if partitions == 1 else None
    if yc is None:
        yc = 0.5 * (y[1:] + y[:-1])
    T_exact = 1.0 + gas.prandtl / (2.0 * gas.cp) * yc * (1.0 - yc)
    res = solver.integrator.residual_norms()
    return {
        "u_err": np.abs(prim["u"] - yc).max(),
        "T_err": np.abs(prim["T"] - T_exact).max() / (T_exact.max() - 1.0),
        "p_var": np.ptp(prim["p"]) / prim["p"].mean(),
        "res": res,
        "steps": solver.step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="8,16,32")
    ap.add_argument("--stretch", type=float, default=0.0)
    args = ap.parse_args()
    init_warp()
    grids = [int(g) for g in args.grids.split(",")]
    prev = None
    print(f"{'ny':>4} {'max|u-ue|':>11} {'max|T-Te|/dT':>13} {'ord':>5} {'p var':>9} {'res rhoE':>9} {'steps':>6}")
    for n in grids:
        r = run_couette(n, stretch=args.stretch)
        o = np.log(prev / r["T_err"]) / np.log(2) if prev else float("nan")
        print(f"{n:4d} {r['u_err']:11.3e} {r['T_err']:13.3e} {o:5.2f} {r['p_var']:9.2e} {r['res'][4]:9.2e} {r['steps']:6d}")
        prev = r["T_err"]


if __name__ == "__main__":
    main()
