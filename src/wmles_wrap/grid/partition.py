"""Domain decomposition: split parent blocks into sub-blocks and balance them over devices."""

from __future__ import annotations

import heapq

import numpy as np

from .block import SubBlock


WALL_BC_TYPES = ("wall", "wall_modeled")


def line_axes(blocks: list[np.ndarray], wall_faces: set[tuple[int, int]] | None = None) -> list[int]:
    """Line (implicit-solve) axis of every parent block.

    A block with wall faces on exactly one axis uses that wall-normal axis. Otherwise (no walls,
    e.g. wake or outer blocks, or walls on several axes) the axis is chosen from the geometry:
    the axis whose cell size relative to the other two, h_a / sqrt(h_b h_c), has the smallest
    median over the cells, i.e. the direction of the high-aspect-ratio cells. Axes that are one
    cell thick (quasi-2D grids) are never line axes. ``wall_faces``: (block, face) pairs.
    """
    out = []
    for b, x in enumerate(blocks):
        nc = [n - 1 for n in x.shape[:3]]
        cand = [a for a in range(3) if nc[a] > 1] or [int(np.argmax(nc))]
        walls = sorted({f // 2 for bb, f in (wall_faces or ()) if bb == b} & set(cand))
        if len(walls) == 1:
            out.append(walls[0])
            continue
        # mean length of the four cell edges along each axis
        h = []
        for a in range(3):
            e = np.linalg.norm(np.diff(x, axis=a), axis=-1)
            for t in range(3):
                if t != a:
                    e = 0.5 * (e[(slice(None),) * t + (slice(None, -1),)] + e[(slice(None),) * t + (slice(1, None),)])
            h.append(np.maximum(e, 1e-300))
        score = {a: float(np.median(h[a] / np.sqrt(np.prod([h[t] for t in range(3) if t != a], axis=0))))
                 for a in (walls or cand)}
        out.append(min(score, key=lambda a: (score[a], a)))
    return out


def split_blocks(blocks: list[np.ndarray], nparts: int, min_cells: int = 4,
                 keep_axes: list[int | None] | None = None) -> list[SubBlock]:
    """Recursively bisect the largest piece along its longest cell direction until there are
    at least ``nparts`` pieces. Pieces are never split below ``min_cells`` cells in a direction,
    and never along ``keep_axes[parent]`` (the line axis, so implicit lines stay whole).
    """
    pieces = []  # (parent, lo, hi)
    for b, x in enumerate(blocks):
        nn = x.shape[:3]
        pieces.append((b, (0, 0, 0), tuple(n - 1 for n in nn)))

    def ncells(p):
        _, lo, hi = p
        return int(np.prod([h - l for l, h in zip(lo, hi)]))

    while len(pieces) < nparts:
        pieces.sort(key=ncells, reverse=True)
        for idx, (b, lo, hi) in enumerate(pieces):
            nc = [h - l for l, h in zip(lo, hi)]
            keep = keep_axes[b] if keep_axes is not None else None
            ax = max((a for a in range(3) if a != keep), key=lambda a: (nc[a], -a))
            if nc[ax] >= 2 * min_cells:
                break
        else:
            break  # nothing splittable left
        mid = lo[ax] + nc[ax] // 2
        hi1 = list(hi)
        hi1[ax] = mid
        lo2 = list(lo)
        lo2[ax] = mid
        pieces[idx : idx + 1] = [(b, lo, tuple(hi1)), (b, tuple(lo2), hi)]

    pieces.sort(key=lambda p: (p[0], p[1][2], p[1][1], p[1][0]))
    subs = []
    for sid, (b, lo, hi) in enumerate(pieces):
        x = blocks[b][lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1].copy()
        subs.append(SubBlock(id=sid, parent=b, lo=lo, hi=hi, x=x, parent_nn=blocks[b].shape[:3]))
    return subs


def assign_devices(subs: list[SubBlock], devices: list[str]) -> dict[str, int]:
    """Longest-processing-time greedy assignment of sub-blocks to devices. Returns load per device."""
    heap = [(0, i, d) for i, d in enumerate(devices)]
    heapq.heapify(heap)
    for s in sorted(subs, key=lambda s: s.ncells, reverse=True):
        load, i, d = heapq.heappop(heap)
        s.device = d
        heapq.heappush(heap, (load + s.ncells, i, d))
    return {d: load for load, _, d in heap}
