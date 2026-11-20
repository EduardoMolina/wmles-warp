"""Ghost-cell (halo) exchange between sub-blocks, possibly on different devices.

For every (donor, receiver) sub-block pair the interface copy lists are concatenated into
one index list. Same-device pairs use a single gather/scatter kernel; cross-device pairs
pack on the donor device, ``wp.copy`` the buffer (peer-to-peer when enabled, host-staged
otherwise) and unpack on the receiver.

The current implementation synchronizes the donor device before each cross-device copy.
Overlapping interior computation with transfers (streams/events) is a later optimization.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from ..grid.block import SubBlock
from ..grid.connectivity import HaloMap


@dataclass
class _Pair:
    donor: int
    recv: int
    stage: int
    donor_device: str
    recv_device: str
    n: int
    donor_idx: wp.array  # on donor device
    recv_idx: wp.array  # on recv device
    recv_idx_on_donor: "wp.array | None"  # same-device fast path
    buffers: dict = field(default_factory=dict)  # dtype -> (donor_buf, recv_buf)


_KERNELS: dict = {}


def _kernels(dtype):
    key = dtype
    if key in _KERNELS:
        return _KERNELS[key]

    @wp.kernel(module="unique")
    def gather(src: wp.array3d(dtype=dtype), idx: wp.array(dtype=wp.vec3i), buf: wp.array(dtype=dtype)):
        t = wp.tid()
        d = idx[t]
        buf[t] = src[d[0], d[1], d[2]]

    @wp.kernel(module="unique")
    def scatter(buf: wp.array(dtype=dtype), idx: wp.array(dtype=wp.vec3i), dst: wp.array3d(dtype=dtype)):
        t = wp.tid()
        r = idx[t]
        dst[r[0], r[1], r[2]] = buf[t]

    @wp.kernel(module="unique")
    def copy_direct(
        src: wp.array3d(dtype=dtype),
        sidx: wp.array(dtype=wp.vec3i),
        ridx: wp.array(dtype=wp.vec3i),
        dst: wp.array3d(dtype=dtype),
    ):
        t = wp.tid()
        d = sidx[t]
        r = ridx[t]
        dst[r[0], r[1], r[2]] = src[d[0], d[1], d[2]]

    _KERNELS[key] = (gather, scatter, copy_direct)
    return _KERNELS[key]


class HaloExchanger:
    def __init__(self, subs: list[SubBlock], maps: list[HaloMap], force_staged: bool = False):
        """``force_staged`` routes same-device pairs through the pack/copy/unpack path used
        between devices (lets the multi-GPU code path be tested on a single device)."""
        self.subs = {s.id: s for s in subs}
        grouped: dict[tuple[int, int, int], list[HaloMap]] = defaultdict(list)
        for m in maps:
            grouped[(m.stage, m.donor, m.recv)].append(m)
        self.pairs: list[_Pair] = []
        for (stage, d, r), ms in sorted(grouped.items()):
            didx = np.concatenate([m.donor_idx for m in ms])
            ridx = np.concatenate([m.recv_idx for m in ms])
            if len(didx) == 0:
                continue
            dd, rd = self.subs[d].device, self.subs[r].device
            self.pairs.append(
                _Pair(
                    donor=d,
                    recv=r,
                    stage=stage,
                    donor_device=dd,
                    recv_device=rd,
                    n=len(didx),
                    donor_idx=wp.array(didx, dtype=wp.vec3i, device=dd),
                    recv_idx=wp.array(ridx, dtype=wp.vec3i, device=rd),
                    recv_idx_on_donor=(
                        wp.array(ridx, dtype=wp.vec3i, device=dd) if dd == rd and not force_staged else None
                    ),
                )
            )

    def exchange(self, field_arrays: dict[int, wp.array], stages: tuple[int, ...] = (0, 1, 2)) -> None:
        """Fill ghost cells of every sub-block array in ``field_arrays`` (sub id -> padded array).

        Stage 0 copies from donor interior cells; stages 1, 2 copy edge/corner ghosts whose donor
        cells are ghosts in 1 or 2 directions (run them after applying BCs, in order)."""
        for stage in stages:
            self._exchange_stage(field_arrays, stage)

    def _exchange_stage(self, field_arrays: dict[int, wp.array], stage: int) -> None:
        for p in self.pairs:
            if p.stage != stage:
                continue
            src, dst = field_arrays[p.donor], field_arrays[p.recv]
            gather, scatter, copy_direct = _kernels(src.dtype)
            if p.recv_idx_on_donor is not None:
                wp.launch(copy_direct, dim=p.n, inputs=[src, p.donor_idx, p.recv_idx_on_donor], outputs=[dst],
                          device=p.donor_device)
                continue
            bufs = p.buffers.get(src.dtype)
            if bufs is None:
                bufs = (
                    wp.empty(p.n, dtype=src.dtype, device=p.donor_device),
                    wp.empty(p.n, dtype=src.dtype, device=p.recv_device),
                )
                p.buffers[src.dtype] = bufs
            wp.launch(gather, dim=p.n, inputs=[src, p.donor_idx], outputs=[bufs[0]], device=p.donor_device)
            wp.synchronize_device(p.donor_device)
            wp.copy(bufs[1], bufs[0])
            wp.launch(scatter, dim=p.n, inputs=[bufs[1], p.recv_idx], outputs=[dst], device=p.recv_device)
