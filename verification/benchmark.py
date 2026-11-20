"""Throughput and multi-GPU scaling benchmark (Taylor-Green vortex, periodic box).

Strong scaling: fixed N^3 grid on 1, 2, 4, 8 devices. Weak scaling: N^3 cells per device.
Reports cell-steps per second (one step = 3 RK stages). Intended for the g6e.48xlarge (8x L40S).

    python verification/benchmark.py --n 256 --devices 1,2,4,8 --precision float32 --steps 50
    python verification/benchmark.py --weak --n 192 --devices 1,2,4,8
    python verification/benchmark.py --n 32 --devices cpu --steps 5          # local smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))
from tgv import make_tgv  # noqa: E402

from wmles_wrap.parallel.devices import init_warp  # noqa: E402


def bench(n, devices, precision, steps, warmup=3, partitions_per_device=1, staged=False):
    s = make_tgv(n, reynolds=1600.0, dissipation="roe", precision=precision, sgs="vreman",
                 partitions=len(devices) * partitions_per_device, devices=devices)
    if staged:
        from wmles_wrap.grid.connectivity import build_halo_maps
        from wmles_wrap.parallel.halo import HaloExchanger

        s.halo = HaloExchanger(s.subs, build_halo_maps(s.subs, s.interfaces, s.ng), force_staged=True)
    s.cfg.run.dt = s.integrator.compute_dt(0.5)
    s.advance(nsteps=warmup)  # includes kernel compilation
    wp.synchronize()
    t0 = time.perf_counter()
    s.advance(nsteps=steps)
    wp.synchronize()
    el = time.perf_counter() - t0
    cells = sum(sb.ncells for sb in s.subs)
    ok = all(np.isfinite(x).all() for x in s.parent_states())
    return cells * steps / el, cells, el, ok, s.peer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--devices", default="1,2,4,8", help="comma list of device counts, or 'cpu'")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--weak", action="store_true")
    ap.add_argument("--staged", action="store_true", help="force pack/copy/unpack halos even on one device")
    args = ap.parse_args()
    init_warp()
    if args.devices == "cpu":
        configs = [["cpu"]]
    else:
        ncuda = wp.get_cuda_device_count()
        configs = [[f"cuda:{i}" for i in range(int(c))] for c in args.devices.split(",") if int(c) <= ncuda]
    base = None
    print(f"{'devices':>8} {'cells':>12} {'Mcell-steps/s':>14} {'per device':>11} {'efficiency':>10}")
    for devs in configs:
        n = args.n
        if args.weak:
            n = int(round(args.n * len(devs) ** (1 / 3)))
        rate, cells, el, ok, peer = bench(n, devs, args.precision, args.steps, staged=args.staged)
        per = rate / len(devs)
        if base is None:
            base = per
        print(f"{len(devs):8d} {cells:12,d} {rate / 1e6:14.2f} {per / 1e6:11.2f} {per / base:10.2f}"
              + ("" if ok else "  NON-FINITE"))
    if configs and configs[-1][0] != "cpu":
        print("peer access:", {f"{a}->{b}": v for (a, b), v in peer.items()})


if __name__ == "__main__":
    main()
