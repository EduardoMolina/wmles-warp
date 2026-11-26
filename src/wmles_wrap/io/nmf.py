"""NASA neutral map file (``.nmf``) reader: block BC patches and 1-to-1 interfaces.

The NMF is the boundary file of the NASA Turbulence Modeling Resource grids (written by the
GEOLAB V2k software). After a header with the block count and the block dimensions, each
line is::

    'name'  B1 F1 S1 E1 S2 E2                         (BC patch)
    'one-to-one'  B1 F1 S1 E1 S2 E2  B2 F2 S1 E1 S2 E2  swap

Blocks, faces and indices are 1-based. NMF faces are 1, 2 = k min/max (S along i, then j),
3, 4 = i min/max (j, then k), and 5, 6 = j min/max (k, then i). On the donor side of a
1-to-1 line, S > E means the direction is reversed, and ``swap`` = true pairs the first
receiver direction with the second donor direction.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np

from ..grid.patches import Patch, PatchError, complete_normal

# NMF face -> (our face, first and second tangential axes)
_FACES = {1: (4, 0, 1), 2: (5, 0, 1), 3: (0, 1, 2), 4: (1, 1, 2), 5: (2, 2, 0), 6: (3, 2, 0)}


class NMFError(ValueError):
    pass


def _lines(text: str) -> list[list[str]]:
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].replace("\\", " ").strip()
        if line:
            out.append(shlex.split(line))
    return out


def _range(nn, face_nmf: int, s1, e1, s2, e2):
    face, t1, t2 = _FACES[face_nmf]
    ax = face // 2
    start = [0, 0, 0]
    end = [0, 0, 0]
    start[ax] = end[ax] = 0 if face % 2 == 0 else nn[ax] - 1
    start[t1], end[t1] = s1 - 1, e1 - 1
    start[t2], end[t2] = s2 - 1, e2 - 1
    return face, t1, t2, np.array(start), np.array(end)


def read_nmf(path: str | Path, block_dims: list[tuple[int, int, int]] | None = None) -> list[Patch]:
    """Read the patches of an NMF file. ``block_dims`` (node counts of the grid blocks) is
    checked against the NMF header when given."""
    rows = _lines(Path(path).read_text())
    try:
        nb = int(rows[0][0])
        dims = {}
        for r in rows[1 : 1 + nb]:
            dims[int(r[0]) - 1] = tuple(int(v) for v in r[1:4])
    except (IndexError, ValueError) as e:
        raise NMFError(f"{path}: bad header ({e})") from None
    if sorted(dims) != list(range(nb)):
        raise NMFError(f"{path}: header lists blocks {sorted(d + 1 for d in dims)}, expected 1..{nb}")
    nn = [dims[b] for b in range(nb)]
    if block_dims is not None:
        given = [tuple(int(v) for v in d) for d in block_dims]
        if given != nn:
            raise NMFError(f"{path}: block dimensions {nn} do not match the grid {given}")

    patches = []
    for r in rows[1 + nb :]:
        name = r[0]
        try:
            v = [int(t) for t in r[1:7]]
        except ValueError:
            raise NMFError(f"{path}: bad line {' '.join(r)}") from None
        b, f = v[0] - 1, v[1]
        if not 0 <= b < nb or f not in _FACES:
            raise NMFError(f"{path}: bad block/face in line {' '.join(r)}")
        face, t1, t2, s, e = _range(nn[b], f, *v[2:6])
        lo, hi = tuple(int(x) for x in np.minimum(s, e)), tuple(int(x) for x in np.maximum(s, e))
        if name.lower() != "one-to-one":
            patches.append(Patch(b, face, lo, hi, names=(name,)))
            continue
        try:
            w = [int(t) for t in r[7:13]]
            swap = r[13].lower() in ("true", "t", "1", "yes")
        except (IndexError, ValueError):
            raise NMFError(f"{path}: bad one-to-one line {' '.join(r)}") from None
        bd, fd = w[0] - 1, w[1]
        if not 0 <= bd < nb or fd not in _FACES:
            raise NMFError(f"{path}: bad donor block/face in line {' '.join(r)}")
        dface, u1, u2, ds, de = _range(nn[bd], fd, *w[2:6])
        # receiver direction t1 pairs with donor u1 (u2 if swapped); the starts S map onto each
        # other, so the sign is + when both ranges run the same way
        pairs = ((t1, u2), (t2, u1)) if swap else ((t1, u1), (t2, u2))
        M = np.zeros((3, 3), dtype=np.int64)
        for a, bax in pairs:
            ra, rd = e[a] - s[a], de[bax] - ds[bax]
            if ra == 0 or rd == 0:
                raise NMFError(f"{path}: degenerate one-to-one range in line {' '.join(r)}")
            if abs(ra) != abs(rd):
                raise NMFError(f"{path}: one-to-one ranges differ in length in line {' '.join(r)}")
            M[bax, a] = 1 if (ra > 0) == (rd > 0) else -1
        try:
            M = complete_normal(M, face, dface)
        except PatchError as err:
            raise NMFError(f"{path}: {err}") from None
        patches.append(Patch(b, face, lo, hi, donor=bd, M=M, A0=s.astype(np.int64), B0=ds.astype(np.int64)))
    return patches
