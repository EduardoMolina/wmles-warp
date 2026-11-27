"""NASA TMR 2D coflowing jet (VERIF/2DCJ, formerly "2D planar shear") with SA: grid convergence of
the plate drag C_D and of the centreline velocity u(x, y=0) at x = 2.71623, 29.2468, 95.501 against
the CFL3D and FUN3D SA results (cases/tmr/2DCJ/convergence.dat, uvel_y_0.dat).

Reference state (CFL3D input): M_ref = 0.5, Re = 5e4 per unit length, T = 540 R (300 K),
nut_inf = 3 nu_inf. Inflow (inflow_total): inner jet below the plate p0/p_ref = 1.1862,
T0/T_ref = 1.05 (= the free-stream totals, M = 0.5), outer stream p0/p_ref = 1.046, T0/T_ref = 1.0
(M = 0.254); back pressure p/p_ref = 1; symmetry on the centreline and the top. The plate
(-10 < x < 0, height 0.5) is a zero-thickness wall with both sides viscous.
C_D: skin friction on both sides, over 1/2 rho_ref U_ref^2 * 10 (CFL3D SREF = 10, per unit span).
u is reported over a_ref (TMR convention) = M_ref * u / U_ref; the centreline value is the first
cell's (symmetric ghost).
The grids (*.p3dfmt + *.nmf) must be in cases/tmr/2DCJ/ (git-ignored). Grid coordinates: x
streamwise, z normal (TMR y), y spanwise.

    conda run -n pyvista env PYTHONPATH=src python verification/sa_jet.py [--levels 4,3,2[,1]]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from wmles_wrap.config import config_from_dict
from wmles_wrap.parallel.devices import init_warp
from wmles_wrap.post.surface import wall_data
from wmles_wrap.solver import Solver

TMR = Path(__file__).resolve().parents[1] / "cases" / "tmr" / "2DCJ"
MACH = 0.5
XS = (2.71623, 29.2468, 95.501)
BCS = [{"patches": ["viscous_solid"], "type": "wall"},
       {"patches": ["symmetry_y_strong", "symmetry_z_strong"], "type": "slip"},
       {"patches": ["subsonic_inflow_pt"], "block": 0, "type": "inflow_total"},  # inner jet: free-stream totals
       {"patches": ["subsonic_inflow_pt"], "block": 1, "type": "inflow_total",
        "params": {"p_total": 1.046, "T_total": 1.0}},
       {"patches": ["back_pressure"], "type": "outflow"}]


def read_reference():
    """({code: {N: (C_D, u1, u2, u3)}}, {code: (x, u)} along y = 0)."""
    conv, code = {}, None
    for line in (TMR / "convergence.dat").read_text().splitlines():
        line = line.strip()
        if line.startswith("zone"):
            code = line.split('"')[1].split(",")[0]
            conv[code] = {}
        elif line and code and line[0].isdigit():
            v = [float(x) for x in line.split()]
            conv[code][int(v[0])] = (v[3], v[4], v[5], v[6])
    prof, code, rows = {}, None, []
    for line in (TMR / "uvel_y_0.dat").read_text().splitlines():
        line = line.strip()
        if line.startswith("zone"):
            if code:
                prof[code] = np.array(rows)
            code, rows = line.split('"')[1], []
        elif line and code and (line[0].isdigit() or line[0] == "-"):
            v = line.split()
            rows.append((float(v[0]), float(v[2])))
    prof[code] = np.array(rows)
    return conv, prof


def centreline(s):
    """x and u/a_ref of the first cell row above the centreline (kmin of the downstream blocks)."""
    xs, us = [], []
    states = s.parent_states()
    for b, x in enumerate(s.parent_blocks):
        lab = s.grid.labels[b]
        if 4 in lab and "symmetry_z_strong" in lab[4] and x[..., 0].min() >= -1e-12:
            W = s.gas.primitives(states[b][0, :, 0])
            xc = 0.25 * (x[0, :-1, 0, 0] + x[0, 1:, 0, 0] + x[0, :-1, 1, 0] + x[0, 1:, 1, 0])
            xs.append(xc)
            us.append(MACH * W["u"])
    x, u = np.concatenate(xs), np.concatenate(us)
    o = np.argsort(x)
    return x[o], u[o]


def run(level: int, partitions: int = 1, max_iterations: int = 3000, rans: dict | None = None):
    grid = f"shear_{level}levelsdown"
    tmp = Path(tempfile.mkdtemp())
    s = Solver(config_from_dict({
        "grid": {"file": f"{grid}.p3dfmt", "nmf": f"{grid}.nmf"},
        "flow": {"mach": MACH, "reynolds": 5e4, "temperature": 300.0},
        "sgs": {"model": "none"}, "wall_model": {"type": "none"}, "numerics": {"flux": "roe_muscl", "limiter": "none"},
        "solver": {"mode": "rans"},
        "rans": {"turbulence": "sa", "max_iterations": max_iterations, "tolerance": 1e-10, **(rans or {})},
        "boundaries": BCS,
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp / "out")},
        "parallel": {"devices": ["cpu"], "partitions": partitions}}, case_dir=TMR))
    t0 = time.perf_counter()
    converged = s.run_steady()
    fx, span = 0.0, None
    for (b, _), wd in wall_data(s).items():
        A = np.linalg.norm(wd.S, axis=-1)
        fx += float(np.sum(wd.tau[..., 0] * A))
        x = s.parent_blocks[b]
        span = float(np.abs(x[1, 0, 0, 1] - x[0, 0, 0, 1]))
    cd = fx / (0.5 * 10.0 * span)
    xc, uc = centreline(s)
    ncells = sum(sb.ncells for sb in s.subs)
    return (ncells, cd, [float(np.interp(x, xc, uc)) for x in XS], (xc, uc), s.step, converged,
            time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4,3,2")
    ap.add_argument("--rans", default="{}", help="extra rans settings (YAML), e.g. '{linear_solver: fgmres}'")
    args = ap.parse_args()
    init_warp()
    conv, prof = read_reference()
    print(f"{'N':>7} {'iters':>5} {'C_D':>11} {'CFL3D':>11} {'FUN3D':>11}  "
          + "  ".join(f"{'u(' + str(x) + ')':>13} {'CFL3D':>8} {'FUN3D':>8}" for x in XS))
    for lev in (int(v) for v in args.levels.split(",")):
        n, cd, us, (xc, uc), it, ok, dt = run(lev, rans=yaml.safe_load(args.rans))
        ref = {c: conv[c].get(n, (np.nan,) * 4) for c in ("CFL3D", "FUN3D")}
        row = f"{n:7d} {it:5d} {cd:11.5e} {ref['CFL3D'][0]:11.5e} {ref['FUN3D'][0]:11.5e}  "
        row += "  ".join(f"{u:13.6f} {ref['CFL3D'][i + 1]:8.6f} {ref['FUN3D'][i + 1]:8.6f}" for i, u in enumerate(us))
        print(row + ("" if ok else "  (not converged)") + f"  {dt:.0f} s")
        for c, arr in prof.items():
            px, pu = arr[:, 0], arr[:, 1]
            sel = (px > xc.min()) & (px < xc.max())
            d = np.interp(px[sel], xc, uc) - pu[sel]
            print(f"        u(x, y=0) vs {c} finest: max |diff| {np.abs(d).max():.4f}, mean {d.mean():+.4f}")
    fine = max(conv["CFL3D"])
    print(f"TMR finest ({fine} cells): C_D {conv['CFL3D'][fine][0]:.5e} / {conv['FUN3D'][fine][0]:.5e}, "
          f"u {conv['CFL3D'][fine][1:]} / {conv['FUN3D'][fine][1:]}")


if __name__ == "__main__":
    main()
