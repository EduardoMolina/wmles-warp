"""NASA TMR 2D zero-pressure-gradient flat plate (VERIF/2DZP) with SA: grid convergence of C_f at
x = 0.97 and of the plate drag against the CFL3D and FUN3D SA results (cases/tmr/2DZP/*.dat).

M = 0.2, Re = 5e6 per unit length, T = 540 R (300 K), nut_inf = 3 nu_inf. TMR boundary conditions:
inflow p0/p_inf = 1.02828, T0/T_inf = 1.008 (inflow_total, the free-stream totals), back pressure
p/p_inf = 1 (outflow), Riemann far field on top. C_f from the wall-normal velocity derivative with
the first two cells (post/surface.py, order 2; --order 1: first cell only);
C_D = (1/2) int_0^2 C_f dx (TMR normalizes by the plate length 2).
The grids (*.p3dfmt + *.nmf) must be in cases/tmr/2DZP/ (git-ignored).

    conda run -n pyvista env PYTHONPATH=src python verification/sa_flatplate.py [--levels 4,3,2,1[,0]] [--order 2]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from wmles_wrap.config import config_from_dict
from wmles_wrap.parallel.devices import init_warp
from wmles_wrap.post.surface import wall_data
from wmles_wrap.solver import Solver

TMR = Path(__file__).resolve().parents[1] / "cases" / "tmr" / "2DZP"
GRIDS = {4: "flatplate_clust2_4levelsdown_35x25", 3: "flatplate_clust2_3levelsdown_69x49",
         2: "flatplate_clust2_2levelsdown_137x97", 1: "flatplate_clust2_1leveldown_273x193",
         0: "grid_struct_545x385_vol"}  # level 0: the finest TMR grid, CGNS only
BCS = {"plot3d": [{"patches": ["viscous_solid"], "type": "wall"},
                  {"patches": ["symmetry_y_strong", "symmetry_z_strong"], "type": "slip"},
                  {"patches": ["subsonic_inflow_pt"], "type": "inflow_total"},
                  {"patches": ["back_pressure"], "type": "outflow"},
                  {"patches": ["farfield_riem"], "type": "farfield"}],
       "cgns": [{"patches": ["BCWallViscousHeatFlux"], "type": "wall"},
                {"patches": ["BCSymmetryPlane", "BCExtrapolate"], "type": "slip"},
                {"patches": ["BCTunnelInflow"], "type": "inflow_total"},
                {"patches": ["BCTunnelOutflow"], "type": "outflow"},
                {"patches": ["BCFarfield"], "type": "farfield"}]}


def read_tmr(name: str) -> dict[str, dict[int, float]]:
    """{code: {N cells: value}} from a TMR Tecplot data file."""
    out, zone = {}, None
    for line in (TMR / name).read_text().splitlines():
        line = line.strip()
        if line.startswith("zone"):
            zone = line.split('"')[1].split(",")[0]
            out[zone] = {}
        elif line and zone and line[0].isdigit():
            v = line.split()
            out[zone][int(float(v[0]))] = float(v[-1])
    return out


def run(level: int, partitions: int = 1, max_iterations: int = 1000, order: int = 2):
    grid = GRIDS[level]
    tmp = Path(tempfile.mkdtemp())
    fmt = "cgns" if level == 0 else "plot3d"
    gcfg = ({"file": f"{grid}.cgns", "format": "cgns"} if fmt == "cgns"
            else {"file": f"{grid}.p3dfmt", "nmf": f"{grid}.nmf"})
    s = Solver(config_from_dict({
        "grid": gcfg,
        "flow": {"mach": 0.2, "reynolds": 5e6, "temperature": 300.0},
        "sgs": {"model": "none"}, "wall_model": {"type": "none"}, "numerics": {"flux": "roe_muscl"},
        "solver": {"mode": "rans"},
        "rans": {"turbulence": "sa", "max_iterations": max_iterations, "tolerance": 1e-10},
        "boundaries": BCS[fmt],
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp / "out")},
        "parallel": {"devices": ["cpu"], "partitions": partitions}}, case_dir=TMR))
    t0 = time.perf_counter()
    converged = s.run_steady()
    (b, _), wd = next(iter(wall_data(s, order=order).items()))  # the plate (kmin of the plate block)
    xs, cf = wd.xf[0, :, 0], wd.cf[0]
    x = s.parent_blocks[b]
    span = np.linalg.norm(x[1, 0, 0] - x[0, 0, 0])  # quasi-2D grid: one cell in i (y)
    cd = 0.5 * float(np.sum(cf * np.linalg.norm(wd.S[0], axis=-1))) / span
    ncells = sum(sb.ncells for sb in s.subs)
    return ncells, float(np.interp(0.970084071, xs, cf)), cd, s.step, converged, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4,3,2,1")
    ap.add_argument("--order", type=int, default=2, help="wall-shear evaluation: 1 first cell, 2 two cells")
    args = ap.parse_args()
    init_warp()
    ref_cf, ref_cd = read_tmr("cf_convergence.dat"), read_tmr("drag_convergence.dat")
    print(f"{'N':>7} {'iters':>5} {'Cf(0.97)':>11} {'CFL3D':>11} {'FUN3D':>11} {'C_D':>11} {'CFL3D':>11} {'FUN3D':>11}")
    for lev in (int(v) for v in args.levels.split(",")):
        n, cf, cd, it, ok, dt = run(lev, order=args.order)
        print(f"{n:7d} {it:5d} {cf:11.5e} {ref_cf['CFL3D'][n]:11.5e} {ref_cf['FUN3D'][n]:11.5e} "
              f"{cd:11.5e} {ref_cd['CFL3D'][n]:11.5e} {ref_cd['FUN3D'][n]:11.5e}" + ("" if ok else "  (not converged)")
              + f"  {dt:.0f} s")
    print(f"TMR finest (208896 cells): Cf {ref_cf['CFL3D'][208896]:.5e} / {ref_cf['FUN3D'][208896]:.5e}, "
          f"C_D {ref_cd['CFL3D'][208896]:.5e} / {ref_cd['FUN3D'][208896]:.5e}")


if __name__ == "__main__":
    main()
