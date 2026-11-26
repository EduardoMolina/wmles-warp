"""Grid input with patch information: formatted Plot3D, NASA neutral map files (NMF), structured
CGNS (HDF5 and ADF), splitting of blocks at patch boundaries, and BC selection by patch label.
The TMR tests use the grids in cases/tmr/ and are skipped when those (git-ignored) files are
missing."""

from pathlib import Path

import numpy as np
import pytest

from cgns_writer import bc_node, cgns_tree, conn_node, write_tree_hdf5, zone_node
from wmles_wrap.config import ConfigError, config_from_dict
from wmles_wrap.grid.patches import Patch, PatchError, split_at_patches
from wmles_wrap.io.cgns import read_cgns, read_tree
from wmles_wrap.io.nmf import read_nmf
from wmles_wrap.io.plot3d import read_plot3d
from wmles_wrap.solver import Solver

TMR = Path(__file__).resolve().parents[1] / "cases" / "tmr"


# ---- formatted Plot3D ------------------------------------------------------------------------

def _write_formatted(path, blocks, multiblock=True, two_d=False, iblank=False, exp="E"):
    vals = []
    if multiblock:
        vals.append([len(blocks)])
    for b in blocks:
        vals.append(list(b.shape[:2] if two_d else b.shape[:3]))
    lines = [" ".join(str(v) for v in row) for row in vals]
    for b in blocks:
        nc = 2 if two_d else 3
        data = np.ascontiguousarray(b[..., :nc].transpose(3, 2, 1, 0)).ravel()
        lines += [" ".join(f"{v:.16E}".replace("E", exp) for v in data[i : i + 3]) for i in range(0, data.size, 3)]
        if iblank:
            lines.append(" ".join("1" for _ in range(b[..., 0].size)))
    Path(path).write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("multiblock", [True, False])
@pytest.mark.parametrize("iblank", [False, True])
@pytest.mark.parametrize("exp", ["E", "D"])
def test_formatted_plot3d(tmp_path, multiblock, iblank, exp):
    rng = np.random.default_rng(2)
    blocks = [rng.standard_normal((4, 3, 2, 3)), rng.standard_normal((2, 5, 3, 3))]
    blocks = blocks if multiblock else blocks[:1]
    _write_formatted(tmp_path / "g.fmt", blocks, multiblock, iblank=iblank, exp=exp)
    back = read_plot3d(tmp_path / "g.fmt")
    assert len(back) == len(blocks)
    for a, b in zip(blocks, back):
        assert np.array_equal(a, b)


def test_formatted_plot3d_2d(tmp_path):
    rng = np.random.default_rng(3)
    b = rng.standard_normal((5, 4, 1, 3))
    b[..., 2] = 0.0
    _write_formatted(tmp_path / "g2.fmt", [b], multiblock=True, two_d=True)
    back = read_plot3d(tmp_path / "g2.fmt")
    assert back[0].shape == (5, 4, 1, 3) and np.array_equal(back[0], b)


# ---- a three-block grid with sub-face BCs and reoriented blocks --------------------------------

def _box(x0, x1, ni=5, nk=7):
    x = np.linspace(x0, x1, ni)
    y = np.array([0.0, 0.2])
    z = np.linspace(0.0, 1.0, nk) ** 1.3
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)


def _three_blocks():
    """Blocks A, B, C along x. B is stored with i <-> k swapped (left-handed). The inflow face of
    A carries two BC patches split at k = 3 (1-based), so the cut has to travel through both
    interfaces to C."""
    A, C = _box(0, 1), _box(2, 3)
    B = np.ascontiguousarray(_box(1, 2).transpose(2, 1, 0, 3))  # (7, 2, 5): i = z, k = x
    sym = lambda n, nn: [bc_node(f"{n}_jlo", "BCSymmetryPlane", (1, 1, 1), (nn[0], 1, nn[2])),  # noqa: E731
                         bc_node(f"{n}_jhi", "BCSymmetryPlane", (1, 2, 1), (nn[0], 2, nn[2]))]
    za = zone_node("A", A, bcs=[
        bc_node("inA_lo", "BCInflow", (1, 1, 1), (1, 2, 3)),
        bc_node("inA_hi", "FamilySpecified", (1, 1, 3), (1, 2, 7), family="INLET"),
        bc_node("wallA", "BCWall", (1, 1, 1), (5, 2, 1)),
        bc_node("farA", "BCFarfield", (1, 1, 7), (5, 2, 7)),
        *sym("A", (5, 2, 7))],
        conns=[conn_node("AB", "B", (5, 1, 1), (5, 2, 7), (1, 1, 1), (7, 2, 1), (3, 2, 1))])
    zb = zone_node("B", B, bcs=[
        bc_node("wallB", "BCWall", (1, 1, 1), (1, 2, 5)),
        bc_node("farB", "BCFarfield", (7, 1, 1), (7, 2, 5)),
        *sym("B", (7, 2, 5))],
        conns=[conn_node("BA", "A", (1, 1, 1), (7, 2, 1), (5, 1, 1), (5, 2, 7), (3, 2, 1)),
               conn_node("BC", "C", (1, 1, 5), (7, 2, 5), (1, 1, 1), (1, 2, 7), (3, 2, 1))])
    zc = zone_node("C", C, bcs=[
        bc_node("outC", "BCOutflow", (5, 1, 1), (5, 2, 7)),
        bc_node("wallC", "BCWall", (1, 1, 1), (5, 2, 1)),
        bc_node("farC", "BCFarfield", (1, 1, 7), (5, 2, 7)),
        *sym("C", (5, 2, 7))],
        conns=[conn_node("CB", "B", (1, 1, 1), (1, 2, 7), (1, 1, 5), (7, 2, 5), (3, 2, 1))])
    return (A, B, C), cgns_tree([za, zb, zc], families={"INLET": "BCInflow"})


def test_cgns_hdf5_read(tmp_path):
    blocks, tree = _three_blocks()
    write_tree_hdf5(tmp_path / "g.cgns", tree)
    got, patches, names = read_cgns(tmp_path / "g.cgns")
    assert names == ["A", "B", "C"]
    for a, b in zip(blocks, got):
        assert np.array_equal(a, b)
    by_name = {p.names[0]: p for p in patches if not p.is_interface}
    assert by_name["inA_hi"].names == ("inA_hi", "INLET", "BCInflow")
    assert by_name["inA_lo"].face == 0 and by_name["inA_lo"].hi == (0, 1, 2)
    assert by_name["wallB"].face == 0 and by_name["farB"].face == 1
    # 1-to-1 transforms map receiver nodes onto coincident donor nodes
    itf = [p for p in patches if p.is_interface]
    assert len(itf) == 4
    for p in itf:
        idx = np.stack(np.meshgrid(*[np.arange(l, h + 1) for l, h in zip(p.lo, p.hi)], indexing="ij"), -1).reshape(-1, 3)
        d = p.map_nodes(idx)
        xr = got[p.block][idx[:, 0], idx[:, 1], idx[:, 2]]
        xd = got[p.donor][d[:, 0], d[:, 1], d[:, 2]]
        assert np.array_equal(xr, xd)


def test_split_propagates_cuts():
    blocks, tree = _three_blocks()
    from wmles_wrap.io import cgns as cg

    # read through the tree interpreter without a file
    orig = cg.read_tree
    cg.read_tree = lambda path: tree
    try:
        got, patches, _ = read_cgns("unused")
    finally:
        cg.read_tree = orig
    g = split_at_patches(got, patches)
    # A cut at k = 2 (0-based); B (i = z) at i = 2; C at k = 2
    assert [x.shape[:3] for x in g.blocks] == [(5, 2, 3), (5, 2, 5), (3, 2, 5), (5, 2, 5), (5, 2, 3), (5, 2, 5)]
    assert g.origin == [(0, (0, 0, 0)), (0, (0, 0, 2)), (1, (0, 0, 0)), (1, (2, 0, 0)), (2, (0, 0, 0)), (2, (0, 0, 2))]
    assert g.labels[0][0] == ("inA_lo", "BCInflow") and g.labels[1][0] == ("inA_hi", "INLET", "BCInflow")
    assert g.interface_faces == {(0, 1), (1, 1), (2, 4), (3, 4), (2, 5), (3, 5), (4, 0), (5, 0)}


def test_overlapping_patches_rejected():
    x = _box(0, 1)
    p = [Patch(0, 4, (0, 0, 0), (4, 1, 0), ("a",)), Patch(0, 4, (2, 0, 0), (4, 1, 0), ("b",))]
    with pytest.raises(PatchError, match="overlapping"):
        split_at_patches([x], p)


def _three_block_config(tmp_path, partitions, **grid):
    return config_from_dict({
        "grid": {"file": "g.cgns", "format": "cgns", **grid},
        "flow": {"mach": 0.2, "reynolds": 1e4, "temperature": 300.0},
        "boundaries": [
            {"patches": ["BCWall"], "type": "wall"},
            {"patches": ["BCSymmetryPlane"], "type": "slip"},
            {"patches": ["BCInflow"], "type": "inflow"},
            {"patches": ["BCOutflow"], "type": "outflow"},
            {"patches": ["BCFarfield"], "type": "farfield"},
            {"patches": ["inA_hi"], "faces": ["imin"], "type": "farfield"},  # later entries win
        ],
        "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "parallel": {"devices": ["cpu"], "partitions": partitions},
    }, case_dir=tmp_path)


@pytest.mark.parametrize("partitions", [1, 3])
def test_solver_setup_from_cgns(tmp_path, partitions):
    _, tree = _three_blocks()
    write_tree_hdf5(tmp_path / "g.cgns", tree)
    s = Solver(_three_block_config(tmp_path, partitions))
    assert len(s.parent_blocks) == 6
    # 4 imported 1-to-1 faces per side + 3 internal cut faces, both sides each
    assert len(s.interfaces) == 2 * (4 + 3)
    assert s.metrics and min(m.min_volume(s.ng) for m in s.metrics.values()) > 0
    types = {}
    for sb in s.subs:
        for f, bc in sb.face_bc.items():
            types.setdefault((sb.parent, f), set()).add(bc.type)
    assert types[(0, 0)] == {"inflow"} and types[(1, 0)] == {"farfield"}
    assert types[(4, 1)] == {"outflow"} and types[(0, 4)] == {"wall"}
    # B is left-handed: after the k flip its x = 1 face (kmin in the file) is kmax
    assert (2, 5) in {(i.recv, i.recv_face) for i in s.interfaces}
    assert s.grid.labels[2][0] == ("wallB", "BCWall")


def test_missing_interface_match_is_an_error(tmp_path):
    blocks, tree = _three_blocks()
    # move block C away: its imported interface with B no longer matches geometrically
    zc = tree.children[1].child("C")
    for c in zc.child("GridCoordinates").children:
        if c.name == "CoordinateZ":
            c.data = c.data + 0.5
    write_tree_hdf5(tmp_path / "g.cgns", tree)
    with pytest.raises(ValueError, match="without a geometric match"):
        Solver(_three_block_config(tmp_path, 1))


def test_boundary_config_validation(tmp_path):
    base = {"grid": {"file": "g.p3d"}, "flow": {"mach": 0.2, "reynolds": 1e4, "temperature": 300.0}}
    with pytest.raises(ConfigError, match="faces and/or patches"):
        config_from_dict({**base, "boundaries": [{"type": "wall"}]})
    with pytest.raises(ConfigError, match="grid.nmf"):
        config_from_dict({**base, "grid": {"file": "g.cgns", "format": "cgns", "nmf": "g.nmf"}})


# ---- NMF ---------------------------------------------------------------------------------------

NMF_TWO_BLOCKS = """# Block#   IDIM   JDIM   KDIM
2
1 2 5 7
2 5 2 4
# Type B1 F1 S1 E1 S2 E2 B2 F2 S1 E1 S2 E2 Swap
'one-to-one' 1 1 1 2 2 5   2 2 4 1 1 2 true
'wall'       1 1 1 2 1 2
'farfield'   1 2 1 2 1 5
"""


def test_nmf_swap_and_reversed_ranges(tmp_path):
    """Block 1 kmin (i 1..2, j 2..5) matches block 2 kmax with the donor directions swapped
    (receiver i <-> donor j) and receiver j reversed onto donor i."""
    (tmp_path / "g.nmf").write_text(NMF_TWO_BLOCKS)
    p = read_nmf(tmp_path / "g.nmf", [(2, 5, 7), (5, 2, 4)])
    itf = next(q for q in p if q.is_interface)
    assert (itf.block, itf.face, itf.donor) == (0, 4, 1)
    assert itf.lo == (0, 1, 0) and itf.hi == (1, 4, 0)
    # S corners map onto each other: receiver (i=1, j=2) -> donor (i=4, j=1)
    assert np.array_equal(itf.map_nodes([0, 1, 0]), [3, 0, 3])
    assert np.array_equal(itf.map_nodes([1, 4, 0]), [0, 1, 3])  # E -> E
    walls = [q for q in p if q.names == ("wall",)]
    assert walls[0].lo == (0, 0, 0) and walls[0].hi == (1, 1, 0)
    with pytest.raises(ValueError, match="do not match"):
        read_nmf(tmp_path / "g.nmf", [(2, 5, 7), (5, 2, 5)])


# ---- NASA TMR grids ----------------------------------------------------------------------------

TMR_CASES = [
    ("2DZP", "flatplate_clust2_4levelsdown_35x25", "grid_struct_35x25_vol"),
    ("2DZP", "flatplate_clust2_3levelsdown_69x49", "grid_struct_69x49_vol"),
    ("2DB", "bump_4levelsdown_89x41", "bump_struct_89x41_vol"),
    ("2DB", "bump_3levelsdown_177x81", "bump_struct_177x81_vol"),
    ("2DCJ", "shear_4levelsdown", "shear_4levelsdown"),
    ("2DCJ", "shear_3levelsdown", "shear_3levelsdown"),
]
WALL_NMF, WALL_CGNS = "viscous_solid", "BCWallViscousHeatFlux"


def _tmr(case, p3d, cg):
    paths = TMR / case / f"{p3d}.p3dfmt", TMR / case / f"{p3d}.nmf", TMR / case / f"{cg}.cgns"
    if not all(p.exists() for p in paths):
        pytest.skip(f"TMR grid files for {case} not present")
    return paths


@pytest.mark.parametrize("case, p3d, cg", TMR_CASES)
def test_tmr_nmf_and_cgns_agree(case, p3d, cg):
    fmt, nmf, cgns = _tmr(case, p3d, cg)
    b1 = read_plot3d(fmt)
    g1 = split_at_patches(b1, read_nmf(nmf, [b.shape[:3] for b in b1]))
    b2, p2, _ = read_cgns(cgns)
    g2 = split_at_patches(b2, p2)
    assert len(b1) == len(b2) and all(np.array_equal(x, y) for x, y in zip(b1, b2))
    assert g1.origin == g2.origin
    assert all(np.array_equal(x, y) for x, y in zip(g1.blocks, g2.blocks))
    assert g1.interface_faces == g2.interface_faces
    walls1 = {(b, f) for b, lab in enumerate(g1.labels) for f, v in lab.items() if WALL_NMF in v}
    walls2 = {(b, f) for b, lab in enumerate(g2.labels) for f, v in lab.items() if WALL_CGNS in v}
    assert walls1 == walls2 and walls1
    # every face carries a BC label or is matched geometrically (imported 1-to-1 faces and the
    # faces of the cuts); imported 1-to-1 faces never carry a label
    from wmles_wrap.grid.connectivity import detect_interfaces, parent_blocks_as_subs

    for g in (g1, g2):
        connected = {(i.recv, i.recv_face)
                     for i in detect_interfaces(parent_blocks_as_subs(g.blocks), bc_faces=g.bc_faces)}
        assert g.interface_faces <= connected
        for b in range(len(g.blocks)):
            for f in range(6):
                assert (f in g.labels[b]) != ((b, f) in connected), (b, f)


def test_tmr_coflowing_jet_plate_stays_a_wall():
    """The 2DCJ plate is a zero-thickness wall: the lower block's kmax and the upper block's kmin
    coincide, but both carry the wall patch and must not be connected."""
    from wmles_wrap.grid.connectivity import detect_interfaces, parent_blocks_as_subs

    fmt, nmf, _ = _tmr(*TMR_CASES[4])
    b = read_plot3d(fmt)
    g = split_at_patches(b, read_nmf(nmf, [x.shape[:3] for x in b]))
    assert g.labels[0][5] == (WALL_NMF,) and g.labels[1][4] == (WALL_NMF,)
    assert np.array_equal(g.blocks[0][:, :, -1], g.blocks[1][:, :, 0])
    subs = parent_blocks_as_subs(g.blocks)
    assert any((i.recv, i.recv_face) == (0, 5) for i in detect_interfaces(subs))
    assert not any((i.recv, i.recv_face) in {(0, 5), (1, 4)} for i in detect_interfaces(subs, bc_faces=g.bc_faces))


def test_tmr_flatplate_wall_geometry():
    fmt, nmf, _ = _tmr(*TMR_CASES[0])
    b = read_plot3d(fmt)
    g = split_at_patches(b, read_nmf(nmf, [x.shape[:3] for x in b]))
    for blk, lab in enumerate(g.labels):
        for f, v in lab.items():
            face = np.take(g.blocks[blk], 0 if f % 2 == 0 else -1, axis=f // 2)
            if WALL_NMF in v:  # plate: z = 0, 0 <= x <= 2
                assert f == 4 and np.all(face[..., 2] == 0.0)
                assert face[..., 0].min() == pytest.approx(0.0, abs=1e-12) and face[..., 0].max() == pytest.approx(2.0)
            if v == ("subsonic_inflow_pt",):
                assert np.allclose(face[..., 0], face[..., 0].min())


def test_tmr_adf_to_hdf5_roundtrip(tmp_path):
    _, _, cgns = _tmr(*TMR_CASES[4])
    write_tree_hdf5(tmp_path / "h.cgns", read_tree(cgns))
    a, pa, na = read_cgns(cgns)
    b, pb, nb = read_cgns(tmp_path / "h.cgns")
    assert na == nb and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert [(p.block, p.face, p.lo, p.hi, p.names, p.donor) for p in pa] == \
        [(p.block, p.face, p.lo, p.hi, p.names, p.donor) for p in pb]


def test_tmr_case_setup():
    _tmr(*TMR_CASES[0])
    from wmles_wrap.config import load_config

    s = Solver(load_config(TMR / "2DZP" / "case.yaml"))
    assert [x.shape[:3] for x in s.parent_blocks] == [(2, 7, 25), (2, 29, 25)]
    assert s.subs[1].face_bc[4].type == "wall" and s.subs[0].face_bc[4].type == "slip"


@pytest.mark.parametrize("bc", ["slip", "wall"])
def test_one_cell_thick_block(tmp_path, bc):
    """Quasi-2D grids one cell thick (the TMR grids): BC ghost layers beyond the first mirror the
    single interior cell, so a uniform flow along the slab stays uniform to round-off."""
    from wmles_wrap.io.plot3d import write_plot3d

    x, y, z = np.linspace(0, 1, 9), np.array([0.0, 0.2]), np.linspace(0, 1, 7) ** 1.5
    write_plot3d(tmp_path / "g.p3d", [np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)])
    cfg = config_from_dict({
        "grid": {"file": "g.p3d"}, "flow": {"mach": 0.25, "reynolds": 5e4, "temperature": 300.0},
        "boundaries": [{"faces": ["jmin", "jmax"], "type": bc}, {"faces": ["kmin", "kmax"], "type": "slip"},
                       {"faces": ["imin", "imax"], "type": "farfield"}],
        "sgs": {"model": "none"}, "wall_model": {"type": "none"}, "run": {"steps": 5},
        "output": {"dir": str(tmp_path / "out")}, "parallel": {"devices": ["cpu"], "partitions": 1}},
        case_dir=tmp_path)
    s = Solver(cfg)
    U0 = s.parent_states()[0].copy()
    s.run()
    U = s.parent_states()[0]
    assert np.isfinite(U).all()
    if bc == "slip":
        assert np.abs(U - U0).max() < 1e-12
