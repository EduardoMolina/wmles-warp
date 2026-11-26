"""Structured CGNS grid reader: zones, ``ZoneBC`` patches and ``GridConnectivity1to1``.

Both CGNS storage formats are read without the CGNS library:
  * HDF5 (current default, e.g. ICEM/ANSA/Pointwise exports) through ``h5py``;
  * ADF (the pre-3.0 default, e.g. the NASA TMR grids) through a small reader of the ADF
    on-disk layout (``_AdfFile``).

Only what the solver needs is interpreted: the first ``CGNSBase_t`` (or a named one), its
structured zones in file order, ``GridCoordinates``, BCs given by ``PointRange`` at vertices,
family names, and 1-to-1 interfaces given by ``PointRange``/``PointRangeDonor``/``Transform``.
Unsupported features (unstructured zones, ``PointList``, general ``GridConnectivity_t``,
links) raise :class:`CGNSError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..grid.patches import Patch, PatchError, face_of_range, transform_from_cgns

ADF_BLOCK = 4096
_NUMPY = {"I4": "i4", "I8": "i8", "U4": "u4", "U8": "u8", "R4": "f4", "R8": "f8", "C1": "u1", "B1": "u1"}


class CGNSError(ValueError):
    pass


@dataclass
class Node:
    """One CGNS tree node. ``data`` is in Fortran index order (shape == CGNS dimensions)."""

    name: str
    label: str
    dtype: str
    data: object = None  # np.ndarray, str (C1) or None (MT)
    children: list["Node"] = field(default_factory=list)

    def child(self, name: str) -> "Node | None":
        return next((c for c in self.children if c.name == name), None)

    def by_label(self, label: str) -> list["Node"]:
        return [c for c in self.children if c.label == label]

    def text(self) -> str:
        return self.data.strip() if isinstance(self.data, str) else ""


# ---- ADF -----------------------------------------------------------------------------------

class _AdfFile:
    """Reader of the ADF on-disk format (ADF_internals.c). All integers and disk pointers are
    hexadecimal ASCII; a pointer is an 8-digit block number and a 4-digit offset in 4096-byte
    blocks. Node headers are 246 bytes (12 dimensions of 8 digits each)."""

    def __init__(self, buf: bytes):
        self.buf = buf
        if b"ADF Database Version" not in buf[:32]:
            raise CGNSError("not an ADF file")
        if buf[96:100] != b"AdF2" or buf[130:134] != b"AdF4":
            raise CGNSError("unsupported ADF header")
        fmt = chr(buf[100])
        if fmt not in "LB":
            raise CGNSError(f"unsupported ADF numeric format {fmt!r} (only IEEE little/big endian)")
        self.endian = "<" if fmt == "L" else ">"
        self.root = self._ptr(134)

    def _hex(self, a: int, n: int) -> int:
        return int(self.buf[a : a + n], 16)

    def _ptr(self, a: int) -> int:
        return self._hex(a, 8) * ADF_BLOCK + self._hex(a + 8, 4)

    def _str(self, a: int, n: int) -> str:
        return self.buf[a : a + n].decode("ascii", errors="replace").rstrip(" \x00")

    def node(self, a: int) -> Node:
        b = self.buf
        if b[a : a + 4] != b"NoDe" or b[a + 242 : a + 246] != b"TaiL":
            raise CGNSError(f"corrupt ADF node at byte {a}")
        name, label = self._str(a + 4, 32), self._str(a + 36, 32)
        nent = self._hex(a + 76, 8)
        sub = self._ptr(a + 84)
        dtype = self._str(a + 96, 32)
        nd = self._hex(a + 128, 2)
        dims = [self._hex(a + 130 + 8 * i, 8) for i in range(nd)]
        nchunks = self._hex(a + 226, 4)
        chunks = self._ptr(a + 230)
        if dtype == "LK":
            raise CGNSError(f"ADF links are not supported (node {name!r})")
        data = None
        if dtype != "MT" and nchunks > 0:
            data = self._data(dtype, dims, nchunks, chunks, name)
        children = [self.node(p) for p in self._children(sub, nent)] if self._hex(a + 68, 8) else []
        return Node(name, label, dtype, data, children)

    def _children(self, a: int, nent: int) -> list[int]:
        if self.buf[a : a + 4] != b"SNTb":
            raise CGNSError(f"corrupt ADF sub-node table at byte {a}")
        out, p = [], a + 16
        for _ in range(nent):
            name = self._str(p, 32)
            if not name.startswith("unused entry"):
                out.append(self._ptr(p + 32))
            p += 44
        return out

    def _chunk(self, a: int) -> bytes:
        if self.buf[a : a + 4] != b"DaTa":
            raise CGNSError(f"corrupt ADF data chunk at byte {a}")
        end = self._ptr(a + 4)
        return self.buf[a + 16 : end]

    def _data(self, dtype: str, dims: list[int], nchunks: int, a: int, name: str):
        if nchunks == 1:
            raw = self._chunk(a)
        else:
            if self.buf[a : a + 4] != b"DCtb":
                raise CGNSError(f"corrupt ADF data-chunk table at byte {a}")
            raw = b"".join(self._chunk(self._ptr(a + 16 + 24 * i)) for i in range(nchunks))
        if dtype not in _NUMPY:
            raise CGNSError(f"unsupported ADF data type {dtype!r} (node {name!r})")
        n = int(np.prod(dims)) if dims else 0
        size = n * np.dtype(_NUMPY[dtype]).itemsize
        if len(raw) < size:
            raise CGNSError(f"ADF node {name!r}: {len(raw)} data bytes, expected {size}")
        if dtype == "C1":
            return raw[:size].decode("ascii", errors="replace").rstrip(" \x00")
        arr = np.frombuffer(raw[:size], dtype=self.endian + _NUMPY[dtype])
        return arr.reshape(dims, order="F")


def _read_adf(path: Path) -> Node:
    f = _AdfFile(path.read_bytes())
    return f.node(f.root)


# ---- HDF5 ----------------------------------------------------------------------------------

def _attr(obj, key: str) -> str:
    v = obj.attrs.get(key, b"")
    if isinstance(v, np.ndarray):
        v = v.tobytes()
    if isinstance(v, bytes):
        v = v.split(b"\x00", 1)[0].decode("ascii", errors="replace")
    return str(v).strip()


def _h5_node(g) -> Node:
    name, label, dtype = _attr(g, "name"), _attr(g, "label"), _attr(g, "type")
    if dtype == "LK":
        raise CGNSError(f"HDF5 CGNS links are not supported (node {name!r})")
    data = None
    if " data" in g:
        arr = g[" data"][()]
        if dtype == "C1":
            data = np.asarray(arr, dtype=np.uint8).tobytes().decode("ascii", errors="replace").rstrip(" \x00")
        else:
            data = np.asarray(arr).T  # HDF5 stores C order with reversed dimensions
    children = [_h5_node(g[k]) for k in g.keys() if not k.startswith(" ")]
    return Node(name, label, dtype, data, children)


def _read_hdf5(path: Path) -> Node:
    try:
        import h5py
    except ImportError:  # pragma: no cover
        raise CGNSError("reading CGNS/HDF5 needs h5py") from None
    with h5py.File(path, "r") as f:
        root = _h5_node(f)
    return root


def read_tree(path: str | Path) -> Node:
    """Read the whole CGNS tree (ADF or HDF5, detected from the file signature)."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(32)
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return _read_hdf5(path)
    if b"ADF Database Version" in head:
        return _read_adf(path)
    raise CGNSError(f"{path}: neither a CGNS/HDF5 nor a CGNS/ADF file")


# ---- CGNS interpretation -------------------------------------------------------------------

def _range(node: Node | None, what: str) -> tuple[np.ndarray, np.ndarray]:
    if node is None or node.data is None:
        raise CGNSError(f"{what}: missing PointRange")
    r = np.asarray(node.data, dtype=np.int64)
    if r.shape != (3, 2):
        raise CGNSError(f"{what}: PointRange has shape {r.shape}, expected (3, 2)")
    return r[:, 0] - 1, r[:, 1] - 1  # begin, end (0-based)


def _families(base: Node) -> dict[str, str]:
    out = {}
    for fam in base.by_label("Family_t"):
        bc = fam.by_label("FamilyBC_t")
        out[fam.name] = bc[0].text() if bc else ""
    return out


def read_cgns(path: str | Path, base: str | None = None):
    """Read a structured CGNS grid.

    Returns ``(blocks, patches, zone_names)``: node coordinates (ni, nj, nk, 3) per zone in
    file order, the BC and 1-to-1 patches (:class:`~wmles_wrap.grid.patches.Patch`), and the
    zone names. BC labels are the BC node name, its BC type, and its family name and family BC
    type when present.
    """
    root = read_tree(path)
    bases = root.by_label("CGNSBase_t")
    if base is not None:
        bases = [b for b in bases if b.name == base]
    if not bases:
        raise CGNSError(f"{path}: no CGNSBase_t" + (f" named {base!r}" if base else ""))
    B = bases[0]
    fams = _families(B)
    zones = B.by_label("Zone_t")
    if not zones:
        raise CGNSError(f"{path}: base {B.name!r} has no zones")
    index = {z.name: n for n, z in enumerate(zones)}
    blocks, patches = [], []
    for zi, z in enumerate(zones):
        zt = z.by_label("ZoneType_t")
        if zt and zt[0].text() != "Structured":
            raise CGNSError(f"zone {z.name!r}: only structured zones are supported ({zt[0].text()})")
        size = np.asarray(z.data, dtype=np.int64)
        if size.ndim != 2 or size.shape[0] != 3:
            raise CGNSError(f"zone {z.name!r}: only 3D structured zones are supported (size {size.tolist()})")
        nn = tuple(int(v) for v in size[:, 0])
        gc = z.child("GridCoordinates") or next(iter(z.by_label("GridCoordinates_t")), None)
        if gc is None:
            raise CGNSError(f"zone {z.name!r}: no GridCoordinates")
        xyz = []
        for c in ("CoordinateX", "CoordinateY", "CoordinateZ"):
            node = gc.child(c)
            if node is None:
                raise CGNSError(f"zone {z.name!r}: missing {c}")
            a = np.asarray(node.data, dtype=np.float64)
            if a.shape != nn:
                raise CGNSError(f"zone {z.name!r}: {c} has shape {a.shape}, expected {nn}")
            xyz.append(a)
        blocks.append(np.ascontiguousarray(np.stack(xyz, axis=-1)))

        for zbc in z.by_label("ZoneBC_t"):
            for bc in zbc.by_label("BC_t"):
                what = f"zone {z.name!r} BC {bc.name!r}"
                loc = bc.by_label("GridLocation_t")
                if loc and loc[0].text() not in ("Vertex", ""):
                    raise CGNSError(f"{what}: GridLocation {loc[0].text()} is not supported (only Vertex)")
                if bc.child("PointList") is not None:
                    raise CGNSError(f"{what}: PointList BCs are not supported")
                s, e = _range(bc.child("PointRange") or next(iter(bc.by_label("IndexRange_t")), None), what)
                lo, hi = np.minimum(s, e), np.maximum(s, e)
                names = [bc.name, bc.text()]
                fn = bc.by_label("FamilyName_t")
                if fn:
                    fam = fn[0].text()
                    names += [fam, fams.get(fam, "")]
                labels = tuple(dict.fromkeys(n for n in names if n and n != "FamilySpecified"))
                try:
                    face = face_of_range(lo, hi, nn)
                except PatchError as err:
                    raise CGNSError(f"{what}: {err}") from None
                patches.append(Patch(zi, face, tuple(int(v) for v in lo), tuple(int(v) for v in hi), labels))

        for zgc in z.by_label("ZoneGridConnectivity_t"):
            if zgc.by_label("GridConnectivity_t"):
                raise CGNSError(f"zone {z.name!r}: general GridConnectivity_t is not supported (only 1-to-1)")
            for c in zgc.by_label("GridConnectivity1to1_t"):
                what = f"zone {z.name!r} interface {c.name!r}"
                donor = c.text()
                if donor not in index:
                    raise CGNSError(f"{what}: unknown donor zone {donor!r}")
                s, e = _range(c.child("PointRange"), what)
                ds, de = _range(c.child("PointRangeDonor"), what)
                t = c.child("Transform")
                T = np.asarray(t.data).ravel().tolist() if t is not None else [1, 2, 3]
                lo, hi = np.minimum(s, e), np.maximum(s, e)
                try:
                    face = face_of_range(lo, hi, nn)
                    M, A0, B0 = transform_from_cgns(T, s, ds)
                except PatchError as err:
                    raise CGNSError(f"{what}: {err}") from None
                if not np.array_equal(M @ (e - s) + ds, de):
                    raise CGNSError(f"{what}: Transform {T} does not map PointRange onto PointRangeDonor")
                patches.append(Patch(zi, face, tuple(int(v) for v in lo), tuple(int(v) for v in hi),
                                     donor=index[donor], M=M, A0=A0, B0=B0))
    return blocks, patches, [z.name for z in zones]
