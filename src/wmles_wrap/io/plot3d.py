"""NASA Plot3D grid reader/writer (pure numpy).

Supports the common variants, detected automatically:
  * formatted (ASCII) files, e.g. the NASA TMR ``.p3dfmt`` grids
  * Fortran unformatted sequential (4-byte record markers) or raw C binary
  * little / big endian
  * single / double precision
  * multi-block or single-block ("whole") files
  * 2D (ni, nj) or 3D (ni, nj, nk) dimensions
  * optional IBLANK arrays (read and discarded)

Blocks are returned as arrays of node coordinates with shape (ni, nj, nk, 3), using
Fortran (i fastest) ordering of the file mapped to numpy index [i, j, k].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Plot3DError(ValueError):
    pass


def _fortran_records(buf: bytes, endian: str) -> list[bytes] | None:
    """Split a Fortran sequential file into records; None if it is not one."""
    recs = []
    pos, n = 0, len(buf)
    while pos < n:
        if pos + 4 > n:
            return None
        (length,) = np.frombuffer(buf, dtype=f"{endian}i4", count=1, offset=pos)
        length = int(length)
        end = pos + 4 + length
        if length < 0 or end + 4 > n:
            return None
        (tail,) = np.frombuffer(buf, dtype=f"{endian}i4", count=1, offset=end)
        if int(tail) != length:
            return None
        recs.append(buf[pos + 4 : end])
        pos = end + 4
    return recs if recs else None


def _decode_block(data: bytes, dims: tuple[int, int, int], endian: str) -> np.ndarray:
    npts = dims[0] * dims[1] * dims[2]
    nbytes = len(data)
    for real, iblank in ((8, False), (4, False), (8, True), (4, True)):
        if nbytes == npts * (3 * real + (4 if iblank else 0)):
            arr = np.frombuffer(data, dtype=f"{endian}f{real}", count=3 * npts)
            arr = arr.reshape(3, dims[2], dims[1], dims[0])  # Fortran order: i fastest
            return np.ascontiguousarray(arr.transpose(3, 2, 1, 0)).astype(np.float64)
    raise Plot3DError(f"block record of {nbytes} bytes does not match dims {dims}")


def _read_fortran(recs: list[bytes], endian: str) -> list[np.ndarray]:
    i4 = f"{endian}i4"
    first = np.frombuffer(recs[0], dtype=i4)
    candidates = []
    # multi-block: [nb] [dims * nb] [blocks...]
    if first.size == 1 and len(recs) >= 2:
        nb = int(first[0])
        dims_all = np.frombuffer(recs[1], dtype=i4)
        for nd in (3, 2):
            if dims_all.size == nb * nd and len(recs) == 2 + nb:
                candidates.append((dims_all.reshape(nb, nd), recs[2:]))
    # single block: [dims] [block]
    for nd in (3, 2):
        if first.size == nd and len(recs) == 2:
            candidates.append((first.reshape(1, nd), recs[1:]))
    for dims_arr, data in candidates:
        try:
            out = []
            for d, rec in zip(dims_arr, data):
                dims = (int(d[0]), int(d[1]), int(d[2]) if d.size == 3 else 1)
                out.append(_decode_block(rec, dims, endian))
            return out
        except Plot3DError:
            continue
    raise Plot3DError("could not interpret Fortran record layout as Plot3D grid")


def _read_raw(buf: bytes, endian: str) -> list[np.ndarray]:
    i4 = f"{endian}i4"
    ints = np.frombuffer(buf[: min(len(buf), 4 * 4096)], dtype=i4)
    layouts = []
    if ints.size >= 1 and 0 < ints[0] < 100000:
        nb = int(ints[0])
        for nd in (3, 2):
            if ints.size >= 1 + nb * nd:
                layouts.append((nb, nd, 4 * (1 + nb * nd), ints[1 : 1 + nb * nd].reshape(nb, nd)))
    for nd in (3, 2):
        if ints.size >= nd:
            layouts.append((1, nd, 4 * nd, ints[:nd].reshape(1, nd)))
    for nb, nd, offset, dims_arr in layouts:
        if np.any(dims_arr <= 0):
            continue
        dims = [(int(d[0]), int(d[1]), int(d[2]) if nd == 3 else 1) for d in dims_arr]
        npts = [a * b * c for a, b, c in dims]
        remaining = len(buf) - offset
        for real, ib in ((8, 0), (4, 0), (8, 4), (4, 4)):
            if remaining == sum(n * (3 * real + ib) for n in npts):
                out, pos = [], offset
                for d, n in zip(dims, npts):
                    size = n * (3 * real + ib)
                    out.append(_decode_block(buf[pos : pos + size], d, endian))
                    pos += size
                return out
    raise Plot3DError("could not interpret raw binary layout as Plot3D grid")


_TEXT_BYTES = frozenset(b"0123456789+-.eEdD \t\r\n")


def _is_formatted(buf: bytes) -> bool:
    head = buf[:4096]
    return len(head) > 0 and all(c in _TEXT_BYTES for c in head)


def _read_formatted(buf: bytes) -> list[np.ndarray]:
    """Formatted (ASCII) Plot3D, whitespace separated, Fortran D exponents allowed. 2D files
    may carry two coordinates (x, y; z = 0) or three."""
    text = buf.decode("ascii").translate(str.maketrans("dD", "eE"))
    try:
        vals = np.array(text.split(), dtype=np.float64)
    except ValueError:
        raise Plot3DError("formatted Plot3D file contains non-numeric tokens") from None

    def ints(a: np.ndarray) -> np.ndarray | None:
        if a.size and np.all(a == np.round(a)) and np.all(a > 0):
            return a.astype(np.int64)
        return None

    layouts = []
    nb = ints(vals[:1])
    if nb is not None:
        for nd in (3, 2):
            d = ints(vals[1 : 1 + int(nb[0]) * nd])
            if d is not None and d.size == int(nb[0]) * nd:
                layouts.append((d.reshape(-1, nd), 1 + d.size))
    for nd in (3, 2):
        d = ints(vals[:nd])
        if d is not None and d.size == nd:
            layouts.append((d.reshape(1, nd), nd))
    for dims_arr, offset in layouts:
        dims = [(int(d[0]), int(d[1]), int(d[2]) if d.size == 3 else 1) for d in dims_arr]
        npts = [a * b * c for a, b, c in dims]
        ncoords = (3,) if dims_arr.shape[1] == 3 else (3, 2)
        for nc in ncoords:
            for ib in (0, 1):
                if vals.size - offset != sum(n * (nc + ib) for n in npts):
                    continue
                out, pos = [], offset
                for d, n in zip(dims, npts):
                    xyz = np.zeros((3, n))
                    xyz[:nc] = vals[pos : pos + nc * n].reshape(nc, n)
                    pos += n * (nc + ib)
                    arr = xyz.reshape(3, d[2], d[1], d[0])  # Fortran order: i fastest
                    out.append(np.ascontiguousarray(arr.transpose(3, 2, 1, 0)))
                return out
    raise Plot3DError("could not interpret formatted Plot3D layout")


def read_plot3d(path: str | Path) -> list[np.ndarray]:
    """Read a Plot3D grid (binary or formatted). Returns a list of (ni, nj, nk, 3) float64 arrays."""
    buf = Path(path).read_bytes()
    if _is_formatted(buf):
        try:
            return _read_formatted(buf)
        except Plot3DError as e:
            raise Plot3DError(f"{path}: {e}") from None
    for endian in ("<", ">"):
        recs = _fortran_records(buf, endian)
        if recs is not None:
            return _read_fortran(recs, endian)
    for endian in ("<", ">"):
        try:
            return _read_raw(buf, endian)
        except Plot3DError:
            continue
    raise Plot3DError(f"{path}: unrecognized Plot3D format")


def write_plot3d(path: str | Path, blocks: list[np.ndarray]) -> None:
    """Write multi-block, 3D, double precision, little-endian Fortran unformatted Plot3D."""

    def record(fh, arr: np.ndarray):
        data = arr.tobytes()
        marker = np.array([len(data)], dtype="<i4").tobytes()
        fh.write(marker + data + marker)

    with open(path, "wb") as fh:
        record(fh, np.array([len(blocks)], dtype="<i4"))
        record(fh, np.array([b.shape[:3] for b in blocks], dtype="<i4").ravel())
        for b in blocks:
            # Fortran order: x(all), y(all), z(all), i fastest
            record(fh, np.ascontiguousarray(b.transpose(3, 2, 1, 0)).astype("<f8"))
