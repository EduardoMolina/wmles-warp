"""Minimal CGNS/HDF5 writer for tests: builds CGNS trees (``wmles_wrap.io.cgns.Node``) and
writes them with the CGNS/HDF5 layout (groups with ``name``/``label``/``type`` attributes,
data in a `` data`` dataset stored with reversed, C-order dimensions)."""

import numpy as np

from wmles_wrap.io.cgns import Node

_DTYPE = {"I4": "<i4", "I8": "<i8", "R4": "<f4", "R8": "<f8"}


def write_tree_hdf5(path, root: Node) -> None:
    import h5py

    def attrs(g, n: Node):
        g.attrs["name"] = np.bytes_(n.name)
        g.attrs["label"] = np.bytes_(n.label)
        g.attrs["type"] = np.bytes_(n.dtype)
        g.attrs["flags"] = np.array([1], dtype="<i4")

    def write(g, n: Node):
        attrs(g, n)
        if isinstance(n.data, str):
            g.create_dataset(" data", data=np.frombuffer(n.data.encode(), dtype=np.int8))
        elif n.data is not None:
            g.create_dataset(" data", data=np.asarray(n.data, dtype=_DTYPE[n.dtype]).T)
        for c in n.children:
            write(g.create_group(c.name, track_order=True), c)

    with h5py.File(path, "w", track_order=True) as f:
        write(f, root)


def _i4(a):
    return np.asarray(a, dtype=np.int32)


def bc_node(name, bctype, begin, end, family=None) -> Node:
    """BC_t with a vertex PointRange; begin/end are 1-based (i, j, k)."""
    ch = [Node("PointRange", "IndexRange_t", "I4", _i4([begin, end]).T)]
    if family:
        ch.append(Node("FamilyName", "FamilyName_t", "C1", family))
    return Node(name, "BC_t", "C1", bctype, ch)


def conn_node(name, donor, begin, end, dbegin, dend, transform) -> Node:
    return Node(name, "GridConnectivity1to1_t", "C1", donor, [
        Node("Transform", '"int[IndexDimension]"', "I4", _i4(transform)),
        Node("PointRange", "IndexRange_t", "I4", _i4([begin, end]).T),
        Node("PointRangeDonor", "IndexRange_t", "I4", _i4([dbegin, dend]).T),
    ])


def zone_node(name, x, bcs=(), conns=()) -> Node:
    nn = np.array(x.shape[:3])
    size = _i4(np.stack([nn, nn - 1, np.zeros(3, int)], axis=1))
    coords = [Node(f"Coordinate{c}", "DataArray_t", "R8", np.asfortranarray(x[..., a]))
              for a, c in enumerate("XYZ")]
    ch = [Node("ZoneType", "ZoneType_t", "C1", "Structured"),
          Node("GridCoordinates", "GridCoordinates_t", "MT", None, coords)]
    if bcs:
        ch.append(Node("ZoneBC", "ZoneBC_t", "MT", None, list(bcs)))
    if conns:
        ch.append(Node("ZoneGridConnectivity", "ZoneGridConnectivity_t", "MT", None, list(conns)))
    return Node(name, "Zone_t", "I4", size, ch)


def cgns_tree(zones, families=None) -> Node:
    """families: {name: bc type}."""
    fam = [Node(n, "Family_t", "MT", None, [Node("FamilyBC", "FamilyBC_t", "C1", t)])
           for n, t in (families or {}).items()]
    base = Node("Base", "CGNSBase_t", "I4", _i4([3, 3]), list(zones) + fam)
    return Node("HDF5 MotherNode", "Root Node of HDF5 File", "MT", None,
                [Node("CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4", np.array([4.2], dtype=np.float32)), base])
