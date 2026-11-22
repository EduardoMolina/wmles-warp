"""Restart files (.npz). Data is stored per *parent* block (interior cells only), so a run can
restart with a different partitioning or number of devices. Optional extra arrays (e.g. the
accumulated statistics sums) are stored alongside under their own keys."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CORE = ("step", "time", "nblocks", "precision")


def write_restart(path: str | Path, step: int, time: float, parent_states: list[np.ndarray],
                  precision: str, extra: dict | None = None) -> None:
    arrays = {f"block{b:04d}": s for b, s in enumerate(parent_states)}
    extra = extra or {}
    clash = set(extra) & (set(arrays) | set(_CORE))
    if clash:
        raise ValueError(f"restart extra keys clash with core keys: {sorted(clash)}")
    np.savez(path, step=step, time=time, nblocks=len(parent_states), precision=precision, **arrays, **extra)


def read_restart(path: str | Path, with_extra: bool = False):
    """Returns (step, time, states) or, with ``with_extra``, (step, time, states, extra)."""
    with np.load(path) as f:
        nb = int(f["nblocks"])
        names = [f"block{b:04d}" for b in range(nb)]
        states = [f[n].astype(np.float64) for n in names]
        out = (int(f["step"]), float(f["time"]), states)
        if not with_extra:
            return out
        extra = {k: f[k] for k in f.files if k not in names and k not in _CORE}
        return out + (extra,)
