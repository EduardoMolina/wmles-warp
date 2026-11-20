"""Block-to-block connectivity.

Interfaces are detected geometrically: two sub-block faces are connected if their node
sets coincide (optionally after a periodic translation) under one of the 8 in-plane
orientations. Each connection is described by a CGNS-like affine index transform

    idx_donor = M @ (idx_recv - A0) + B0          (logical node indices)

where M is a signed permutation matrix. The transform extends naturally past the face,
so it maps the receiver's ghost nodes/cells onto the donor's interior. Cell indices are
mapped through half-integer node-space positions (c + 1/2).

Connectivity is detected between *parent* blocks (full-face to full-face matches, as in
face-matched Plot3D grids); sub-block halo maps are derived from it, so partitioning never
has to re-detect partially overlapping faces. Partial-face patches
(e.g. from CGNS ZoneGridConnectivity) will be added with the CGNS reader.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .block import SubBlock, face_axis, face_side


@dataclass
class Interface:
    recv: int  # sub-block id receiving ghost data
    recv_face: int
    donor: int
    donor_face: int
    M: np.ndarray  # (3, 3) int signed permutation
    A0: np.ndarray  # (3,) logical node index in recv
    B0: np.ndarray  # (3,) logical node index in donor
    offset: np.ndarray  # (3,) x_donor = x_recv + offset (nonzero for periodic interfaces)

    @property
    def periodic(self) -> bool:
        return bool(np.any(self.offset != 0.0))

    def map_nodes(self, idx: np.ndarray) -> np.ndarray:
        """Map recv logical node indices (..., 3) to donor logical node indices."""
        return (idx - self.A0) @ self.M.T + self.B0

    def map_cells(self, idx: np.ndarray) -> np.ndarray:
        """Map recv logical cell indices (..., 3) to donor logical cell indices."""
        d = (2 * idx + 1 - 2 * self.A0) @ self.M.T + 2 * self.B0
        return (d - 1) // 2


def _tangential(ax: int) -> tuple[int, int]:
    t = [a for a in range(3) if a != ax]
    return t[0], t[1]


def _match_faces(A: SubBlock, fa: int, B: SubBlock, fb: int, offset: np.ndarray, rtol: float):
    FA = A.face_nodes(fa) + offset
    FB = B.face_nodes(fb)
    if FA.size != FB.size:
        return None
    scale = max(np.ptp(FA.reshape(-1, 3), axis=0).max(), 1e-300)
    tol = rtol * scale
    if np.abs(FA.reshape(-1, 3).mean(0) - FB.reshape(-1, 3).mean(0)).max() > tol * 10 + 1e-12:
        return None
    for swap, flip_u, flip_v in itertools.product((False, True), repeat=3):
        G = FB.transpose(1, 0, 2) if swap else FB
        if G.shape != FA.shape:
            continue
        if flip_u:
            G = G[::-1]
        if flip_v:
            G = G[:, ::-1]
        if np.abs(G - FA).max() <= tol:
            return swap, flip_u, flip_v
    return None


def _build_interface(A: SubBlock, fa: int, B: SubBlock, fb: int, orient, offset) -> Interface:
    swap, flip_u, flip_v = orient
    an, bn = face_axis(fa), face_axis(fb)
    a1, a2 = _tangential(an)
    b1, b2 = _tangential(bn)
    bu, bv = (b2, b1) if swap else (b1, b2)
    su, sv = (-1 if flip_u else 1), (-1 if flip_v else 1)
    M = np.zeros((3, 3), dtype=np.int64)
    M[bu, a1] = su
    M[bv, a2] = sv
    d_out_a = 1 if face_side(fa) == 1 else -1
    d_in_b = 1 if face_side(fb) == 0 else -1
    M[bn, an] = d_out_a * d_in_b
    A0 = np.zeros(3, dtype=np.int64)
    A0[an] = 0 if face_side(fa) == 0 else A.nn[an] - 1
    B0 = np.zeros(3, dtype=np.int64)
    B0[bu] = 0 if su == 1 else B.nn[bu] - 1
    B0[bv] = 0 if sv == 1 else B.nn[bv] - 1
    B0[bn] = 0 if face_side(fb) == 0 else B.nn[bn] - 1
    return Interface(A.id, fa, B.id, fb, M, A0, B0, np.asarray(offset, dtype=np.float64))


def detect_interfaces(
    subs: list[SubBlock], translations: list[list[float]] | None = None, rtol: float = 1e-7
) -> list[Interface]:
    """Find all face-to-face connections (direct and periodic). Each physical interface
    yields two entries, one per receiving side."""
    offsets = [np.zeros(3)]
    for t in translations or []:
        t = np.asarray(t, dtype=np.float64)
        offsets += [t, -t]
    out: list[Interface] = []
    for A in subs:
        for fa in range(6):
            found = None
            for B in subs:
                for fb in range(6):
                    for k, off in enumerate(offsets):
                        if A.id == B.id and fa == fb:
                            continue
                        if k == 0 and A.id == B.id and face_axis(fa) == face_axis(fb):
                            # opposite faces of one block can only touch through periodicity
                            continue
                        orient = _match_faces(A, fa, B, fb, off, rtol)
                        if orient is None:
                            continue
                        if found is not None:
                            raise ValueError(
                                f"sub-block {A.id} face {fa} matches several faces "
                                f"({found.donor}:{found.donor_face} and {B.id}:{fb})"
                            )
                        found = _build_interface(A, fa, B, fb, orient, off)
            if found is not None:
                out.append(found)
    return out


# ---- ghost geometry and halo index maps ---------------------------------------------------

def _ghost_slab(nn_or_nc: tuple[int, int, int], face: int, ng: int) -> np.ndarray:
    """Logical indices (N, 3) of the ghost slab beyond ``face`` spanning the full padded
    tangential range. ``nn_or_nc`` are the logical extents (nodes or cells); for nodes the
    face plane itself is shared (interior), so the slab starts one node beyond it."""
    ax, side = face_axis(face), face_side(face)
    ranges = []
    for a in range(3):
        n = nn_or_nc[a]
        if a == ax:
            ranges.append(np.arange(-ng, 0) if side == 0 else np.arange(n, n + ng))
        else:
            ranges.append(np.arange(-ng, n + ng))
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def _in_range(idx: np.ndarray, extents, ng: int) -> np.ndarray:
    ext = np.asarray(extents)
    return np.all((idx >= -ng) & (idx < ext + ng), axis=-1)


def build_padded_nodes(subs: list[SubBlock], interfaces: list[Interface], ng: int) -> None:
    """Fill ``sub.xp`` (padded node coordinates) for parent-level blocks. Physical-boundary ghosts use point mirroring
    through the boundary node; interface ghosts are copied from the donor (minus periodic
    offset), which makes metrics continuous across interfaces."""
    for s in subs:
        nn = s.nn
        xp = np.zeros(tuple(n + 2 * ng for n in nn) + (3,))
        xp[ng : ng + nn[0], ng : ng + nn[1], ng : ng + nn[2]] = s.x
        for ax in range(3):
            n = nn[ax]
            for m in range(1, ng + 1):
                lo_face = np.take(xp, ng, axis=ax)
                hi_face = np.take(xp, ng + n - 1, axis=ax)
                sl = [slice(None)] * 3
                sl[ax] = ng - m
                xp[tuple(sl)] = 2 * lo_face - np.take(xp, ng + m, axis=ax)
                sl[ax] = ng + n - 1 + m
                xp[tuple(sl)] = 2 * hi_face - np.take(xp, ng + n - 1 - m, axis=ax)
        s.xp = xp
    by_id = {s.id: s for s in subs}
    for _ in range(2):  # second pass propagates donor ghosts into corner regions
        for itf in interfaces:
            R, D = by_id[itf.recv], by_id[itf.donor]
            idx = _ghost_slab(R.nn, itf.recv_face, ng)
            didx = itf.map_nodes(idx)
            ok = _in_range(didx, D.nn, ng)
            idx, didx = idx[ok] + ng, didx[ok] + ng
            R.xp[idx[:, 0], idx[:, 1], idx[:, 2]] = D.xp[didx[:, 0], didx[:, 1], didx[:, 2]] - itf.offset


def parent_blocks_as_subs(blocks: list[np.ndarray]) -> list[SubBlock]:
    """Wrap whole parent blocks as SubBlocks (id == parent index) for parent-level connectivity."""
    return [
        SubBlock(id=b, parent=b, lo=(0, 0, 0), hi=tuple(n - 1 for n in x.shape[:3]), x=x, parent_nn=x.shape[:3])
        for b, x in enumerate(blocks)
    ]


def slice_padded_nodes(subs: list[SubBlock], parents: list[SubBlock], ng: int) -> None:
    """Sub-block padded nodes are windows of the parent padded nodes (parent padded index =
    logical + ng), so ghost geometry across internal partition faces is the neighbour's."""
    for s in subs:
        lo, hi = s.lo, s.hi
        s.xp = np.ascontiguousarray(
            parents[s.parent].xp[lo[0] : hi[0] + 1 + 2 * ng, lo[1] : hi[1] + 1 + 2 * ng, lo[2] : hi[2] + 1 + 2 * ng]
        )


@dataclass
class HaloMap:
    """Ghost-cell copy list from one donor sub-block to one receiver sub-block (padded indices)."""

    recv: int
    donor: int
    recv_idx: np.ndarray  # (N, 3) int32 padded indices in recv
    donor_idx: np.ndarray  # (N, 3) int32 padded indices in donor


def build_halo_maps(subs: list[SubBlock], interfaces: list[Interface], ng: int) -> list[HaloMap]:
    """Ghost-cell copy lists between sub-blocks.

    ``interfaces`` are parent-level connections. For every sub-block face that is either an
    internal partition face or lies on a connected parent face, the ghost slab (full padded
    tangential extent) is mapped to donor *parent* cell indices; each donor cell is then
    owned by the donor sub-block whose interior contains it, or, for edge/corner cells beyond
    a physical boundary, by a sub-block whose padded range contains it (BC-filled ghosts).
    """
    itf_by_face = {(i.recv, i.recv_face): i for i in interfaces}
    by_parent: dict[int, list[SubBlock]] = {}
    for s in subs:
        by_parent.setdefault(s.parent, []).append(s)

    pieces: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for s in subs:
        lo_s = np.asarray(s.lo)
        for f in range(6):
            slab = _ghost_slab(s.nc, f, ng)
            gpar = slab + lo_s
            if s.on_parent_face(f):
                itf = itf_by_face.get((s.parent, f))
                if itf is None:
                    continue  # physical boundary
                dpar, dparent = itf.map_cells(gpar), itf.donor
            else:
                dpar, dparent = gpar, s.parent
            todo = np.ones(len(slab), dtype=bool)
            for pad in (0, ng):  # interior ownership first, then padded (ghost) ownership
                for t in by_parent[dparent]:
                    lo_t = np.asarray(t.lo)
                    hi_t = np.asarray(t.hi)  # node index; cells lo..hi-1
                    inside = todo & np.all((dpar >= lo_t - pad) & (dpar < hi_t + pad), axis=1)
                    if pad and t.id == s.id:
                        inside &= False  # never copy own ghosts onto own ghosts
                    if not inside.any():
                        continue
                    pieces.setdefault((s.id, t.id), []).append(
                        (slab[inside] + ng, dpar[inside] - lo_t + ng)
                    )
                    todo &= ~inside
    maps = []
    for (r, d), lst in sorted(pieces.items()):
        maps.append(
            HaloMap(
                recv=r,
                donor=d,
                recv_idx=np.concatenate([a for a, _ in lst]).astype(np.int32),
                donor_idx=np.concatenate([b for _, b in lst]).astype(np.int32),
            )
        )
    return maps


def assign_boundary_conditions(subs: list[SubBlock], interfaces: list[Interface], boundaries) -> None:
    """Attach a BoundaryCfg to every sub-block face that lies on an unconnected parent face.
    ``interfaces`` are parent-level connections."""
    from ..config import FACE_NAMES

    connected = {(i.recv, i.recv_face) for i in interfaces}
    missing = []
    for s in subs:
        s.face_bc = {}
        for f in range(6):
            if not s.on_parent_face(f) or (s.parent, f) in connected:
                continue
            bc = None
            for b in boundaries:
                blocks = b.block if isinstance(b.block, list) else ([b.block] if b.block is not None else None)
                if FACE_NAMES[f] in b.faces and (blocks is None or s.parent in blocks):
                    bc = b
            if bc is None:
                missing.append(f"block {s.parent} {FACE_NAMES[f]}")
            s.face_bc[f] = bc
    if missing:
        raise ValueError("faces without boundary condition or interface: " + ", ".join(sorted(set(missing))))
