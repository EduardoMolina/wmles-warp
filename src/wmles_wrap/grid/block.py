"""Computational (sub-)blocks.

Index conventions used throughout the code:
  * logical node indices 0..nn-1, logical cell indices 0..nc-1 (nc = nn - 1);
    cell c lies between nodes c and c+1.
  * device arrays are padded with ``ng`` ghost layers: padded index = logical + ng.
  * faces are numbered 0..5 = imin, imax, jmin, jmax, kmin, kmax; axis = f // 2, side = f % 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import FACE_NAMES


def face_axis(face: int) -> int:
    return face // 2


def face_side(face: int) -> int:
    return face % 2


def face_id(name: str) -> int:
    return FACE_NAMES.index(name)


@dataclass
class SubBlock:
    """A piece of a parent grid block, the unit of work assigned to one device."""

    id: int
    parent: int
    lo: tuple[int, int, int]  # node index range in the parent (inclusive)
    hi: tuple[int, int, int]
    x: np.ndarray  # (nni, nnj, nnk, 3) node coordinates (no ghosts)
    parent_nn: tuple[int, int, int]
    device: str = "cpu"
    xp: np.ndarray | None = None  # padded node coordinates (filled by connectivity)
    face_bc: dict[int, object] = field(default_factory=dict)

    @property
    def nn(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.x.shape[:3])

    @property
    def nc(self) -> tuple[int, int, int]:
        return tuple(n - 1 for n in self.nn)

    @property
    def ncells(self) -> int:
        a, b, c = self.nc
        return a * b * c

    def on_parent_face(self, face: int) -> bool:
        """True if this sub-block face lies on the corresponding face of its parent block."""
        ax, side = face_axis(face), face_side(face)
        return self.lo[ax] == 0 if side == 0 else self.hi[ax] == self.parent_nn[ax] - 1

    def face_nodes(self, face: int) -> np.ndarray:
        """Node coordinates on a face as a 2D array (n_t1, n_t2, 3), t1 < t2 the tangential axes."""
        ax, side = face_axis(face), face_side(face)
        idx = 0 if side == 0 else self.nn[ax] - 1
        return np.take(self.x, idx, axis=ax)

    def padded_shape_cells(self, ng: int) -> tuple[int, int, int]:
        return tuple(n + 2 * ng for n in self.nc)

    def padded_shape_nodes(self, ng: int) -> tuple[int, int, int]:
        return tuple(n + 2 * ng for n in self.nn)
