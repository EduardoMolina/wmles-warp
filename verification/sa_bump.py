"""NASA TMR 2D bump-in-channel (VERIF/2DB) with SA: grid convergence of C_f at x = 0.6321975, 0.75
and 0.8678025 against the CFL3D and FUN3D SA results (cases/tmr/2DB/cf_convergence_sa.dat).

M = 0.2, Re = 3e6 per unit length, T = 540 R (300 K), nut_inf = 3 nu_inf. TMR boundary conditions:
inflow p0/p_inf = 1.02828, T0/T_inf = 1.008 (inflow_total, the free-stream totals), back pressure
p/p_inf = 1 (outflow), symmetry on top and upstream/downstream of the bump wall (0 <= x <= 1.5).
C_f from the two-cell wall derivative (post/surface.py), signed by the x direction.
The grids (*.p3dfmt + *.nmf; level 1 is CGNS) must be in cases/tmr/2DB/ (git-ignored).

    conda run -n pyvista env PYTHONPATH=src python verification/sa_bump.py [--levels 4,3,2[,1]]
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

TMR = Path(__file__).resolve().parents[1] / "cases" / "tmr" / "2DB"
GRIDS = {4: "bump_4levelsdown_89x41", 3: "bump_3levelsdown_177x81", 2: "bump_2levelsdown_353x161",
         1: "bump_struct_705x321_vol"}  # level 1: CGNS only
XS = (0.6321975, 0.75, 0.8678025)
BCS = {"plot3d": [{"patches": ["viscous_solid"], "type": "wall"},
                  {"patches": ["symmetry_y_strong", "symmetry_z_strong"], "type": "slip"},
                  {"patches": ["subsonic_inflow_pt"], "type": "inflow_total"},
                  {"patches": ["back_pressure"], "type": "outflow"}],
       "cgns": [{"patches": ["BCWallViscousHeatFlux"], "type": "wall"},
                {"patches": ["BCSymmetryPlane", "BCExtrapolate"], "type": "slip"},
                {"patches": ["BCTunnelInflow"], "type": "inflow_total"},
                {"patches": ["BCTunnelOutflow"], "type": "outflow"}]}


def read_reference() -> dict[tuple[str, float], dict[int, float]]:
    """{(code, x): {N cells: C_f}} from cf_convergence_sa.dat."""
    out, key = {}, None
    for line in (TMR / "cf_convergence_sa.dat").read_text().splitlines():
        line = line.strip()
        if line.startswith("zone"):
            code, x = line.split('"')[1].split(", x=")
            key = (code, float(x))
            out[key] = {}
        elif line and key and line[0].isdigit():
            v = line.split()
            out[key][int(float(v[0]))] = float(v[-1])
    return out


def run(level: int, partitions: int = 1, max_iterations: int = 1500):
    grid = GRIDS[level]
    fmt = "cgns" if level == 1 else "plot3d"
    gcfg = ({"file": f"{grid}.cgns", "format": "cgns"} if fmt == "cgns"
            else {"file": f"{grid}.p3dfmt", "nmf": f"{grid}.nmf"})
    tmp = Path(tempfile.mkdtemp())
    s = Solver(config_from_dict({
        "grid": gcfg,
        "flow": {"mach": 0.2, "reynolds": 3e6, "temperature": 300.0},
        "sgs": {"model": "none"}, "wall_model": {"type": "none"}, "numerics": {"flux": "roe_muscl"},
        "solver": {"mode": "rans"},
        "rans": {"turbulence": "sa", "max_iterations": max_iterations, "tolerance": 1e-10},
        "boundaries": BCS[fmt],
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp / "out")},
        "parallel": {"devices": ["cpu"], "partitions": partitions}}, case_dir=TMR))
    t0 = time.perf_counter()
    converged = s.run_steady()
    (_, wd), = wall_data(s).items()
    xs, cf, cp = wd.xf[0, :, 0], wd.cf[0], wd.cp[0]
    ncells = sum(sb.ncells for sb in s.subs)
    return (ncells, [float(np.interp(x, xs, cf)) for x in XS], (xs, cf, cp), s.step, converged,
            time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4,3,2")
    args = ap.parse_args()
    init_warp()
    ref = read_reference()
    print(f"{'N':>7} {'iters':>5} " + " ".join(f"{'Cf(' + str(x)[:6] + ')':>12} {'CFL3D':>10} {'FUN3D':>10}"
                                             for x in XS))
    for lev in (int(v) for v in args.levels.split(",")):
        n, cfs, _, it, ok, dt = run(lev)
        row = f"{n:7d} {it:5d} "
        for x, c in zip(XS, cfs):
            row += f"{c:12.5e} {ref[('CFL3D', x)].get(n, np.nan):10.5e} {ref[('FUN3D', x)].get(n, np.nan):10.5e} "
        print(row + ("" if ok else " (not converged)") + f" {dt:.0f} s")
    print("TMR finest (901120 cells): " + ", ".join(
        f"x={x}: {ref[('CFL3D', x)][901120]:.5e} / {ref[('FUN3D', x)][901120]:.5e}" for x in XS))


if __name__ == "__main__":
    main()
