"""Domain decomposition: split parent blocks into sub-blocks and balance them over devices."""

from __future__ import annotations

import heapq

import numpy as np

from .block import SubBlock


def split_blocks(blocks: list[np.ndarray], nparts: int, min_cells: int = 4) -> list[SubBlock]:
    """Recursively bisect the largest piece along its longest cell direction until there are
    at least ``nparts`` pieces. Pieces are never split below ``min_cells`` cells in a direction.
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
            ax = int(np.argmax(nc))
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
