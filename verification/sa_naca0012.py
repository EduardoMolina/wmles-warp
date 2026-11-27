"""NASA TMR 2D NACA 0012 (validation case, numerical comparison, SA without point vortex):
M = 0.15, Re_c = 6e6, T = 540 R (300 K), alpha = 10 deg, Riemann far field at ~500 c, nut_inf = 3 nu_inf.
Grid family II (cases/tmr/2DNACA0012/n0012familyII.N.p3dfmt + .nmf, git-ignored; single-block C-grid
with the wake cut as a 1-to-1 self-interface). C_L, C_D (pressure + viscous) and C_m about c/4 against
CFL3D (SA, 1st-order turbulence advection like ours; N = cells), FUN3D and TAU (SA-neg, 2nd order;
N = nodes) on the same grids (cases/tmr/2DNACA0012/*_results_sa_nopv_withN.dat).

The grid has z normal to the chord (y spanwise), so the angle of attack is a rotation in the x-z plane:
flow.beta = alpha, free stream (cos a, 0, sin a). Lift along (-sin a, 0, cos a), C_m nose-up positive
(+y moment with x downstream, z up), forces over 1/2 rho U^2 c per unit span.

    conda run -n pyvista env PYTHONPATH=src python verification/sa_naca0012.py [--levels 7,6,5[,4,3]]
"""

from __future__ import annotations

import argparse
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from wmles_wrap.config import config_from_dict
from wmles_wrap.parallel.devices import init_warp
from wmles_wrap.post.surface import wall_data
from wmles_wrap.solver import Solver

TMR = Path(__file__).resolve().parents[1] / "cases" / "tmr" / "2DNACA0012"
ALPHA = 10.0
BCS = [{"patches": ["viscous_solid"], "type": "wall"},
       {"patches": ["symmetry_y_strong"], "type": "slip"},
       {"patches": ["farfield_riem"], "type": "farfield"}]
KEYS = ("CL", "CD", "CDp", "CDv", "CMy")


def read_reference() -> dict[str, dict[int, dict[str, float]]]:
    """{code: {N: {CL, CD, CDp, CDv, CMy}}} for family II (CFL3D: cells; FUN3D/TAU: nodes)."""
    out = {}
    for code, fname in (("CFL3D", "cfl3d_results_sa_nopv_withN.dat"),
                        ("CFL3D-2nd", "cfl3d_results_sa_nopv_2ndorderturbadvection_withN.dat"),
                        ("FUN3D", "fun3d_results_sa_nopv_withN.dat"), ("TAU", "tau_results_sa_nopv_withN.dat")):
        names, zone = None, None
        for line in (TMR / fname).read_text().splitlines():
            s = line.strip()
            if s.upper().startswith("VARIABLES"):
                names = re.findall(r'"([^"]*)"', s)
            elif s.lower().startswith("zone"):
                zone = "Family II," in s
                if zone:
                    out[code] = {}
            elif zone and s and s[0].isdigit():
                v = [float(x) for x in s.split()]
                out[code][int(v[0])] = {k: v[names.index(k)] for k in KEYS}
    return out


def forces(s, alpha=ALPHA):
    """C_L, C_D, C_Dp, C_Dv, C_m(c/4) per unit span from the wall data (chord 1)."""
    gas = s.gas
    p_inf = gas.R  # rho = T = 1
    a = np.radians(alpha)
    drag, lift = np.array([np.cos(a), 0.0, np.sin(a)]), np.array([-np.sin(a), 0.0, np.cos(a)])
    Fp, Fv, My, span = np.zeros(3), np.zeros(3), 0.0, None
    for (b, _), wd in wall_data(s).items():
        A = np.linalg.norm(wd.S, axis=-1)[..., None]
        fp = -(wd.p - p_inf)[..., None] * wd.S
        fv = wd.tau * A
        Fp += fp.reshape(-1, 3).sum(0)
        Fv += fv.reshape(-1, 3).sum(0)
        r = wd.xf - np.array([0.25, 0.0, 0.0])
        My += float(np.cross(r, fp + fv)[..., 1].sum())
        x = s.parent_blocks[b]
        span = float(np.abs(x[1, 0, 0, 1] - x[0, 0, 0, 1]))
    q = 0.5 * span
    F = Fp + Fv
    return {"CL": F @ lift / q, "CD": F @ drag / q, "CDp": Fp @ drag / q, "CDv": Fv @ drag / q, "CMy": My / q}


def run(level: int, partitions: int = 1, max_iterations: int = 3000, rans: dict | None = None):
    grid = f"n0012familyII.{level}"
    tmp = Path(tempfile.mkdtemp())
    s = Solver(config_from_dict({
        "grid": {"file": f"{grid}.p3dfmt", "nmf": f"{grid}.nmf"},
        "flow": {"mach": 0.15, "reynolds": 6e6, "temperature": 300.0, "beta": ALPHA},
        "sgs": {"model": "none"}, "wall_model": {"type": "none"}, "numerics": {"flux": "roe_muscl", "limiter": "none"},
        "solver": {"mode": "rans"},
        "rans": {"turbulence": "sa", "max_iterations": max_iterations, "tolerance": 1e-10, **(rans or {})},
        "boundaries": BCS,
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp / "out")},
        "parallel": {"devices": ["cpu"], "partitions": partitions}}, case_dir=TMR))
    t0 = time.perf_counter()
    converged = s.run_steady()
    x = s.parent_blocks
    ncells = sum(sb.ncells for sb in s.subs)
    nj = sum(b.shape[1] for b in x) - (len(x) - 1)  # C-grid nodes around (blocks cut at the wake)
    nodes = nj * x[0].shape[2]
    return ncells, nodes, forces(s), s.step, converged, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="7,6,5")
    ap.add_argument("--rans", default="{}", help="extra rans settings (YAML), e.g. '{linear_solver: fgmres}'")
    args = ap.parse_args()
    init_warp()
    ref = read_reference()
    for lev in (int(v) for v in args.levels.split(",")):
        n, nodes, c, it, ok, dt = run(lev, rans=yaml.safe_load(args.rans))
        print(f"family II.{lev}: {n} cells, {it} iterations{'' if ok else ' (not converged)'}, {dt:.0f} s")
        print(f"  {'':10s}" + "".join(f"{k:>12s}" for k in KEYS))
        print(f"  {'ours':10s}" + "".join(f"{c[k]:12.6f}" for k in KEYS))
        for code in ("CFL3D", "CFL3D-2nd", "FUN3D", "TAU"):
            r = ref.get(code, {}).get(n if code.startswith("CFL3D") else nodes)
            if r:
                print(f"  {code:10s}" + "".join(f"{r[k]:12.6f}" for k in KEYS))
    print("TMR infinite-grid bounds (2nd-order turbulence advection): CL 1.0909-1.0911, CD 0.012270-0.012275, "
          "CDp 0.006060-0.006070, CDv 0.006205-0.006206, CM 0.00676-0.00680")


if __name__ == "__main__":
    main()
