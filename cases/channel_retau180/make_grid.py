"""Wall-resolved LES grid for turbulent channel flow at Re_tau = 180 (Moser, Kim & Mansour 1999).

Domain 2 pi h x 2h x pi h, uniform in x/z, tanh clustering in y with first-cell y+ ~ 1.
    conda run -n pyvista env PYTHONPATH=src python cases/channel_retau180/make_grid.py [--nx 32 --ny 64 --nz 32]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from wmles_wrap.io.plot3d import write_plot3d


def tanh_y(ny: int, dy1: float) -> np.ndarray:
    eta = np.linspace(0.0, 1.0, ny + 1)

    def y_of(g):
        return np.tanh(g * (2 * eta - 1)) / np.tanh(g)

    g = brentq(lambda g: (y_of(g)[1] - y_of(g)[0]) - dy1, 1e-3, 10.0)
    return y_of(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--retau", type=float, default=180.0)
    ap.add_argument("--yplus1", type=float, default=1.0)
    args = ap.parse_args()
    y = tanh_y(args.ny, args.yplus1 / args.retau)
    x = np.linspace(0, 2 * np.pi, args.nx + 1)
    z = np.linspace(0, np.pi, args.nz + 1)
    g = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    out = Path(__file__).parent / f"channel_retau180_{args.nx}x{args.ny}x{args.nz}.p3d"
    write_plot3d(out, [g])
    dyc = np.diff(y)
    print(f"wrote {out}: dx+ {2 * np.pi / args.nx * args.retau:.1f}, dz+ {np.pi / args.nz * args.retau:.1f}, "
          f"dy+ wall {dyc[0] * args.retau:.2f}, centre {dyc[args.ny // 2] * args.retau:.1f}")


if __name__ == "__main__":
    main()
