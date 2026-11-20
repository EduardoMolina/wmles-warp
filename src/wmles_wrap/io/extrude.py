"""Extrude 2D (nk == 1) Plot3D blocks in z. Python port of cases/channel/extrude.f90."""

from __future__ import annotations

import numpy as np


def extrude_block(block: np.ndarray, nk: int, length: float) -> np.ndarray:
    """Extrude a (ni, nj, 1, 3) block to (ni, nj, nk, 3) with uniform spacing length/(nk-1)."""
    if block.ndim != 4 or block.shape[2] != 1:
        raise ValueError(f"expected a single 2D plane (ni, nj, 1, 3), got {block.shape}")
    if nk < 2:
        raise ValueError("nk must be >= 2")
    dz = length / (nk - 1)
    out = np.repeat(block, nk, axis=2)
    out[..., 2] = block[..., 2] + np.arange(nk)[None, None, :] * dz
    return out


def channel_nk(ni: int, aspect_ratio: int = 1, lx: float = 12.8, lz: float = 4.8) -> int:
    """Spanwise node count used by the UMD channel grids (dx/dz = aspect_ratio)."""
    nk = int(round((ni - 1) * lz / lx)) + 1
    return aspect_ratio * (nk - 1) + 1
