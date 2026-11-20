"""Shared synthetic grids/cases for tests."""

import numpy as np

from wmles_wrap.config import config_from_dict
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.solver import Solver

LX, LY, LZ = 2.0, 1.0, 1.0


def _perturbed_box(ni, nj, nk, amp=0.04):
    """Curvilinear grid on [0,LX]x[0,LY]x[0,LZ]; periodic in x,z, flat walls at y=0,LY."""
    x, y, z = np.meshgrid(np.linspace(0, LX, ni), np.linspace(0, LY, nj), np.linspace(0, LZ, nk), indexing="ij")
    s = np.sin(2 * np.pi * x / LX) * np.sin(2 * np.pi * z / LZ) * np.sin(np.pi * y / LY)
    c = np.cos(2 * np.pi * x / LX) * np.sin(np.pi * y / LY) * np.sin(2 * np.pi * z / LZ)
    return np.stack([x + amp * s, y + amp * c * 0.5, z + amp * s * c], axis=-1)


def _field(xc):
    """Smooth function periodic in x and z (so periodic ghost values are well defined)."""
    x, y, z = xc[..., 0], xc[..., 1], xc[..., 2]
    return np.sin(2 * np.pi * x / LX) * np.cos(2 * np.pi * z / LZ) + y**2 + 0.3 * np.cos(2 * np.pi * x / LX) * y


def _two_block_case(tmp_path, partitions, flip_handedness=False, extra=None, dims=(17, 9, 11)):
    g = _perturbed_box(*dims)
    h = dims[0] // 2
    a, b = g[: h + 1], g[h:]  # split along x, shared node plane at i = 8
    # reorient block 1: cyclic axis permutation (i,j,k) -> (j,k,i) and flip two axes
    b = b.transpose(1, 2, 0, 3)[::-1, :, ::-1]
    if flip_handedness:
        b = b[:, ::-1]  # left-handed -> solver must reverse k
    write_plot3d(tmp_path / "g.p3d", [np.ascontiguousarray(a), np.ascontiguousarray(b)])
    data = {
            "grid": {"file": "g.p3d"},
            "flow": {"mach": 0.3, "reynolds": 1e4, "temperature": 300.0},
            "boundaries": [
                {"faces": ["jmin", "jmax"], "type": "wall", "block": 0},
                {"faces": ["imin", "imax"], "type": "wall", "block": 1},
            ],
            "periodic": [{"translation": [LX, 0, 0]}, {"translation": [0, 0, LZ]}],
            "parallel": {"devices": ["cpu"], "partitions": partitions},
        }
    for k, v in (extra or {}).items():
        data[k] = {**data[k], **v} if isinstance(v, dict) and isinstance(data.get(k), dict) else v
    return Solver(config_from_dict(data, case_dir=tmp_path))
