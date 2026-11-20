"""Taylor-Green vortex on [0, 2 pi]^3 (periodic).

Phase 1 use: inviscid kinetic-energy conservation of the KEEP-PE scheme (dissipation off) and
the amount of dissipation the shock sensor adds in vortical flow (should be ~zero).
Phase 2 use: viscous TGV at Re = 1600, M = 0.1 (dissipation rate vs DNS) via --re.

    conda run -n pyvista env PYTHONPATH=src python verification/tgv.py --n 16 --t-end 5 [--dissipation roe]
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


def make_tgv(n=16, mach=0.1, reynolds=None, dissipation="none", sensor="ducros_pressure", partitions=1,
             precision="float64", cfl=0.8, sgs="none", devices=("cpu",)):
    tmp = Path(tempfile.mkdtemp())
    L = 2 * np.pi
    x = np.stack(np.meshgrid(*(np.linspace(0, L, n + 1),) * 3, indexing="ij"), axis=-1)
    write_plot3d(tmp / "tgv.p3d", [x])
    cfg = config_from_dict(
        {
            "grid": {"file": "tgv.p3d"},
            "flow": {"mach": mach, "reynolds": float(reynolds or 1.0), "temperature": 300.0},
            "gas": {"viscosity": "constant"},
            "numerics": {"viscous": reynolds is not None, "dissipation": dissipation, "sensor": sensor,
                         "precision": precision, "cfl": cfl},
            "sgs": {"model": sgs},
            "periodic": [{"translation": t} for t in ([L, 0, 0], [0, L, 0], [0, 0, L])],
            "parallel": {"devices": list(devices), "partitions": partitions},
        },
        case_dir=tmp,
    )
    solver = Solver(cfg)
    R = solver.gas.R

    def init(xc):
        X, Y, Z = xc[..., 0], xc[..., 1], xc[..., 2]
        p = R + (np.cos(2 * X) + np.cos(2 * Y)) * (np.cos(2 * Z) + 2.0) / 16.0
        W = np.zeros(xc.shape[:-1] + (5,))
        W[..., 0] = p / R  # T = 1
        W[..., 1] = np.sin(X) * np.cos(Y) * np.cos(Z)
        W[..., 2] = -np.cos(X) * np.sin(Y) * np.cos(Z)
        W[..., 4] = p
        return W

    solver.set_primitive_field(init)
    return solver


def kinetic_energy(solver) -> float:
    """Volume-averaged kinetic energy per unit volume, sum rho |u|^2 / 2 V / sum V."""
    ke, vol = 0.0, 0.0
    for s in solver.subs:
        U = solver.interior(solver.U[s.id].numpy()).astype(np.float64)
        v = solver.interior(solver.metrics[s.id].vol.numpy()).astype(np.float64)
        ke += (0.5 * (U[..., 1] ** 2 + U[..., 2] ** 2 + U[..., 3] ** 2) / U[..., 0] * v).sum()
        vol += v.sum()
    return ke / vol


# DNS reference for Re = 1600 (van Rees et al., J. Comput. Phys. 2011; Brachet et al. 1983):
# peak dissipation rate eps = -dk/dt ~ 0.0127 at t ~ 9.0 (time in L/V0, k per unit mass and volume)
DNS_PEAK_EPS = 0.0127
DNS_PEAK_T = 9.0


def run_tgv_dissipation(n=32, reynolds=1600.0, sgs="vreman", t_end=20.0, sample_dt=0.1, mach=0.1,
                        dissipation="roe", partitions=1, precision="float64", log=False):
    """Returns times, k(t), eps(t) = -dk/dt (central differences at sample midpoints)."""
    s = make_tgv(n, mach=mach, reynolds=reynolds, dissipation=dissipation, partitions=partitions,
                 precision=precision, sgs=sgs)
    times, ks = [0.0], [kinetic_energy(s)]
    for t in np.arange(sample_dt, t_end + 0.5 * sample_dt, sample_dt):
        s.advance(t_end=float(t))
        times.append(s.time)
        ks.append(kinetic_energy(s))
        if log and len(times) % 20 == 0:
            print(f"  t={s.time:6.2f} k={ks[-1]:.6f} step={s.step}", flush=True)
    times, ks = np.array(times), np.array(ks)
    eps = -np.diff(ks) / np.diff(times)
    tm = 0.5 * (times[1:] + times[:-1])
    return times, ks, tm, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--t-end", type=float, default=5.0)
    ap.add_argument("--dissipation", default="none")
    ap.add_argument("--re", type=float, default=None, help="viscous TGV (e.g. 1600); default inviscid")
    ap.add_argument("--sgs", default="none")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()
    init_warp()
    if args.re is not None:
        times, ks, tm, eps = run_tgv_dissipation(args.n, args.re, args.sgs, args.t_end,
                                                 dissipation=args.dissipation, log=True)
        out = Path(__file__).parent / "results" / f"tgv_re{int(args.re)}_n{args.n}_{args.sgs}_{args.dissipation}.csv"
        out.parent.mkdir(exist_ok=True)
        np.savetxt(out, np.column_stack([tm, eps, 0.5 * (ks[1:] + ks[:-1])]), delimiter=",",
                   header="t,eps,k", comments="")
        i = int(np.argmax(eps))
        print(f"TGV Re={args.re:g} n={args.n} sgs={args.sgs} diss={args.dissipation}: peak eps {eps[i]:.5f} at t={tm[i]:.2f}"
              f"  (DNS {DNS_PEAK_EPS} at t~{DNS_PEAK_T}); wrote {out}")
        return
    s = make_tgv(args.n, dissipation=args.dissipation)
    k0 = kinetic_energy(s)
    print(f"{'t':>8} {'k/k0':>14} {'(k-k0)/k0':>12}")
    for t in np.linspace(0, args.t_end, args.samples + 1)[1:]:
        s.advance(t_end=t)
        k = kinetic_energy(s)
        print(f"{s.time:8.3f} {k / k0:14.10f} {(k - k0) / k0:12.3e}")


if __name__ == "__main__":
    main()
