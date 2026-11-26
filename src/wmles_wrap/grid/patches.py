"""Boundary and interface patches from structured grid files (CGNS, NASA neutral map file),
and splitting of blocks so that every patch covers a full block face.

The solver attaches BCs and detects interfaces per full parent face. Grid files instead
describe sub-face patches: a BC on part of a face, or a 1-to-1 interface between parts of two
faces. ``split_at_patches`` cuts every block at every patch boundary and carries the cuts
through the 1-to-1 interfaces to the donor blocks until no new cut appears. Afterwards each
block face is covered by exactly one patch. Interfaces are then found by the existing geometric
full-face matching, and the imported 1-to-1 list only checks that result.

Index conventions: node indices are 0-based and ranges are inclusive. Faces are numbered
0..5 = imin, imax, jmin, jmax, kmin, kmax, as in :mod:`wmles_wrap.grid.block`. A 1-to-1
interface maps receiver node indices to donor node indices with the CGNS-like transform
``idx_d = M @ (idx_r - A0) + B0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import FACE_NAMES


class PatchError(ValueError):
    pass


@dataclass
class Patch:
    block: int
    face: int
    lo: tuple[int, int, int]  # inclusive node range; lo[axis] == hi[axis] == face plane
    hi: tuple[int, int, int]
    names: tuple[str, ...] = ()  # BC labels (empty for interfaces)
    donor: int | None = None  # donor block of a 1-to-1 interface
    M: np.ndarray | None = None  # (3, 3) signed permutation
    A0: np.ndarray | None = None  # (3,) receiver node index
    B0: np.ndarray | None = None  # (3,) donor node index

    @property
    def is_interface(self) -> bool:
        return self.donor is not None

    def map_nodes(self, idx: np.ndarray) -> np.ndarray:
        return (np.asarray(idx) - self.A0) @ self.M.T + self.B0


@dataclass
class SplitGrid:
    """Blocks after splitting, with the patch labels of every block face."""

    blocks: list[np.ndarray]
    labels: list[dict[int, tuple[str, ...]]]  # per block: face -> BC labels (BC faces only)
    interface_faces: set[tuple[int, int]] = field(default_factory=set)  # (block, face) on 1-to-1 patches
    origin: list[tuple[int, tuple[int, int, int]]] = field(default_factory=list)  # (file block, node offset)

    @property
    def bc_faces(self) -> set[tuple[int, int]]:
        """(block, face) of the faces with a BC patch."""
        return {(b, f) for b, lab in enumerate(self.labels) for f in lab}

    def flip_k(self, b: int) -> None:
        """Reverse the k index of block ``b`` (left-handed grids), swapping kmin/kmax data."""
        self.blocks[b] = np.ascontiguousarray(self.blocks[b][:, :, ::-1])
        lab = self.labels[b]
        lo, hi = lab.pop(4, None), lab.pop(5, None)
        if lo is not None:
            lab[5] = lo
        if hi is not None:
            lab[4] = hi
        faces = {(bb, f) for bb, f in self.interface_faces if bb == b and f in (4, 5)}
        self.interface_faces -= faces
        self.interface_faces |= {(bb, 9 - f) for bb, f in faces}


def face_of_range(lo, hi, nn) -> int:
    """Face number of an inclusive node range lying on a block face."""
    for ax in range(3):
        if lo[ax] == hi[ax] and lo[ax] in (0, nn[ax] - 1):
            other = [a for a in range(3) if a != ax]
            if all(hi[a] > lo[a] for a in other):
                return 2 * ax + (0 if lo[ax] == 0 else 1)
    raise PatchError(f"range {lo}..{hi} is not a face patch of a block with {nn} nodes")


def tangential(face: int) -> tuple[int, int]:
    ax = face // 2
    t = [a for a in range(3) if a != ax]
    return t[0], t[1]


def transform_from_cgns(T, begin_r, begin_d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CGNS ``Transform`` (1-based signed axes) and range starts (0-based) -> (M, A0, B0)."""
    M = np.zeros((3, 3), dtype=np.int64)
    for a, t in enumerate(T):
        t = int(t)
        if t == 0 or abs(t) > 3:
            raise PatchError(f"invalid Transform {list(T)}")
        M[abs(t) - 1, a] = 1 if t > 0 else -1
    if sorted(np.abs(M).sum(axis=0)) != [1, 1, 1] or sorted(np.abs(M).sum(axis=1)) != [1, 1, 1]:
        raise PatchError(f"Transform {list(T)} is not a permutation")
    return M, np.asarray(begin_r, dtype=np.int64), np.asarray(begin_d, dtype=np.int64)


def complete_normal(M: np.ndarray, face_r: int, face_d: int) -> np.ndarray:
    """Fill the normal-axis entry of M (outward on the receiver = inward on the donor)."""
    M = M.copy()
    an, bn = face_r // 2, face_d // 2
    M[:, an] = 0
    M[bn, an] = (1 if face_r % 2 == 1 else -1) * (1 if face_d % 2 == 0 else -1)
    return M


def _inverse(p: Patch, donor_nn) -> Patch:
    Mi = p.M.T.copy()
    c = np.array([p.lo, p.hi])
    dc = p.map_nodes(c)
    lo, hi = tuple(int(v) for v in dc.min(0)), tuple(int(v) for v in dc.max(0))
    return Patch(p.donor, face_of_range(lo, hi, donor_nn), lo, hi, (), p.block, Mi, p.B0.copy(), p.A0.copy())


def _check(blocks, patches) -> None:
    for p in patches:
        if not 0 <= p.block < len(blocks):
            raise PatchError(f"patch on block {p.block + 1}: the grid has {len(blocks)} blocks")
        nn = blocks[p.block].shape[:3]
        if any(l < 0 or h >= n or l > h for l, h, n in zip(p.lo, p.hi, nn)):
            raise PatchError(f"patch on block {p.block + 1}: range {p.lo}..{p.hi} outside nodes {nn}")
        if face_of_range(p.lo, p.hi, nn) != p.face:
            raise PatchError(f"patch on block {p.block + 1}: range {p.lo}..{p.hi} is not on {FACE_NAMES[p.face]}")
        if p.is_interface:
            if not 0 <= p.donor < len(blocks):
                raise PatchError(f"interface on block {p.block + 1}: unknown donor block {p.donor + 1}")
            dnn = blocks[p.donor].shape[:3]
            dc = p.map_nodes(np.array([p.lo, p.hi]))
            if np.any(dc < 0) or np.any(dc >= np.asarray(dnn)):
                raise PatchError(f"interface block {p.block + 1} -> {p.donor + 1}: donor range outside nodes {dnn}")


def split_at_patches(blocks: list[np.ndarray], patches: list[Patch]) -> SplitGrid:
    """Split ``blocks`` so that every patch covers whole faces of the new blocks.

    New blocks are numbered in the order of the file blocks; the pieces of one file block are
    ordered with i fastest, then j, then k. A file block that needs no cut keeps its shape.
    """
    _check(blocks, patches)
    cuts = [[{0, n - 1} for n in x.shape[:3]] for x in blocks]
    for p in patches:
        for a in tangential(p.face):
            cuts[p.block][a] |= {p.lo[a], p.hi[a]}
    links = []
    for p in patches:
        if p.is_interface:
            links += [p, _inverse(p, blocks[p.donor].shape[:3])]
    changed = True
    while changed:
        changed = False
        for L in links:
            for a in tangential(L.face):
                b = int(np.flatnonzero(L.M[:, a])[0])
                for c in sorted(cuts[L.block][a]):
                    if L.lo[a] <= c <= L.hi[a]:
                        d = int(L.B0[b] + L.M[b, a] * (c - L.A0[a]))
                        if d not in cuts[L.donor][b]:
                            cuts[L.donor][b].add(d)
                            changed = True

    out = SplitGrid(blocks=[], labels=[])
    for fb, x in enumerate(blocks):
        nn = x.shape[:3]
        iv = [sorted(c) for c in cuts[fb]]
        iv = [list(zip(c[:-1], c[1:])) for c in iv]
        for k0, k1 in iv[2]:
            for j0, j1 in iv[1]:
                for i0, i1 in iv[0]:
                    lo, hi = (i0, j0, k0), (i1, j1, k1)
                    nb = len(out.blocks)
                    out.blocks.append(np.ascontiguousarray(x[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]))
                    out.origin.append((fb, lo))
                    out.labels.append({})
                    for f in range(6):
                        ax, side = f // 2, f % 2
                        if (side == 0 and lo[ax] != 0) or (side == 1 and hi[ax] != nn[ax] - 1):
                            continue  # internal cut face
                        flo, fhi = list(lo), list(hi)
                        flo[ax] = fhi[ax] = 0 if side == 0 else nn[ax] - 1
                        t = tangential(f)

                        def covers(p):
                            return p.block == fb and p.face == f and all(
                                p.lo[a] <= flo[a] and fhi[a] <= p.hi[a] for a in t)

                        bcs = [p for p in patches if not p.is_interface and covers(p)]
                        # 1-to-1 patches count from both sides (an NMF lists each interface once)
                        itf = any(covers(L) for L in links)
                        if len(bcs) + itf > 1:
                            raise PatchError(f"block {fb + 1} {FACE_NAMES[f]}: overlapping patches over "
                                             f"nodes {tuple(flo)}..{tuple(fhi)}")
                        if itf:
                            out.interface_faces.add((nb, f))
                        elif bcs:
                            out.labels[nb][f] = bcs[0].names
                        # else uncovered: the case file must give a BC by face name
    return out


def unsplit(blocks: list[np.ndarray]) -> SplitGrid:
    """Wrap blocks read without patch information."""
    return SplitGrid(blocks=list(blocks), labels=[{} for _ in blocks],
                     origin=[(b, (0, 0, 0)) for b in range(len(blocks))])
