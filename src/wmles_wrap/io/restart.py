"""Restart files (.npz). Data is stored per *parent* block (interior cells only), so a run can
restart with a different partitioning or number of devices."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_restart(path: str | Path, step: int, time: float, parent_states: list[np.ndarray],
                  precision: str) -> None:
    arrays = {f"block{b:04d}": s for b, s in enumerate(parent_states)}
    np.savez(path, step=step, time=time, nblocks=len(parent_states), precision=precision, **arrays)


def read_restart(path: str | Path) -> tuple[int, float, list[np.ndarray]]:
    with np.load(path) as f:
        nb = int(f["nblocks"])
        states = [f[f"block{b:04d}"].astype(np.float64) for b in range(nb)]
        return int(f["step"]), float(f["time"]), states
