"""Field output with PyVista: one MultiBlock (.vtm) of StructuredGrids per snapshot, plus a
ParaView .pvd collection file indexing the snapshots in time."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
import pyvista as pv


def structured_grid(x: np.ndarray, cell_data: dict[str, np.ndarray] | None = None) -> pv.StructuredGrid:
    """x: (ni, nj, nk, 3) nodes; cell_data arrays: (ni-1, nj-1, nk-1[, ncomp])."""
    grid = pv.StructuredGrid(x[..., 0], x[..., 1], x[..., 2])
    for name, arr in (cell_data or {}).items():
        if arr.ndim == 3:
            grid.cell_data[name] = arr.ravel(order="F")
        else:
            grid.cell_data[name] = arr.reshape(-1, arr.shape[-1], order="F")
    return grid


class SnapshotWriter:
    def __init__(self, out_dir: str | Path, basename: str = "flow"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.basename = basename
        self.entries: list[tuple[float, str]] = []

    def write(self, step: int, time: float, blocks: list[tuple[str, np.ndarray, dict]]) -> Path:
        mb = pv.MultiBlock()
        for name, x, data in blocks:
            mb.append(structured_grid(x, data), name)
        fname = f"{self.basename}_{step:08d}.vtm"
        path = self.dir / fname
        mb.save(path)
        self.entries.append((time, fname))
        self._write_pvd()
        return path

    def _write_pvd(self) -> None:
        lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "  <Collection>"]
        for t, f in self.entries:
            lines.append(f'    <DataSet timestep="{t:.10g}" part="0" file={quoteattr(f)}/>')
        lines += ["  </Collection>", "</VTKFile>", ""]
        (self.dir / f"{self.basename}.pvd").write_text("\n".join(lines))
