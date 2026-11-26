"""Wall distance (grid/wall_distance.py): exact analytic distances (flat plate, faceted cylinder),
a numpy brute-force reference on a curvilinear multiblock grid, float32 storage far from the
origin, partition independence, and grids without walls."""

from pathlib import Path

import numpy as np
import pytest

from helpers import _two_block_case
from wmles_wrap.config import config_from_dict, load_config
from wmles_wrap.grid.wall_distance import NO_WALL_DISTANCE, wall_surface
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.solver import Solver

TMR = Path(__file__).resolve().parents[1] / "cases" / "tmr"


def brute_force_distance(P, V, T):
    """Independent reference: plane projection if inside the triangle, else nearest edge."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    out = np.empty(len(P))
    for k in range(0, len(P), 256):
        p = P[k : k + 256, None, :]
        h = np.sum((p - a) * n, -1)
        q = p - h[..., None] * n
        inside = np.ones(h.shape, dtype=bool)
        for u, v in ((a, b), (b, c), (c, a)):
            inside &= np.sum(np.cross(v - u, q - u) * n, -1) >= 0
        d = np.where(inside, np.abs(h), np.inf)
        for u, v in ((a, b), (b, c), (c, a)):
            e = v - u
            t = np.clip(np.sum((p - u) * e, -1) / np.sum(e * e, -1), 0.0, 1.0)
            d = np.minimum(d, np.linalg.norm(p - (u + t[..., None] * e), axis=-1))
        out[k : k + 256] = d.min(1)
    return out


def _interior(solver, sid, a):
    g = solver.ng
    return a[g:-g, g:-g, g:-g]


def _centroids64(s, ng):
    x = s.xp
    xc = 0.125 * sum(x[i : x.shape[0] - 1 + i, j : x.shape[1] - 1 + j, k : x.shape[2] - 1 + k]
                     for i in (0, 1) for j in (0, 1) for k in (0, 1))
    return xc[ng:-ng, ng:-ng, ng:-ng]


def _plate_case(tmp_path, precision, offset=(0.0, 0.0, 0.0), partitions=1):
    """Flat plate z = 0 for 0 <= x <= 2 behind a slip run-in (-0.5 <= x < 0); first cell 1e-6."""
    z = np.concatenate([[0.0], 1e-6 * np.cumsum(1.25 ** np.arange(40))])
    z = z[z <= 1.0]
    y = np.array([0.0, 0.1])
    up = np.stack(np.meshgrid(np.linspace(-0.5, 0.0, 6), y, z, indexing="ij"), -1) + offset
    down = np.stack(np.meshgrid(np.linspace(0.0, 2.0, 11) ** 1.5 / 2.0 ** 0.5, y, z, indexing="ij"), -1) + offset
    write_plot3d(tmp_path / "plate.p3d", [up, down])
    return Solver(config_from_dict({
        "grid": {"file": "plate.p3d"},
        "flow": {"mach": 0.2, "reynolds": 5e6, "temperature": 300.0},
        "numerics": {"precision": precision},
        "boundaries": [
            {"faces": ["kmin"], "block": 0, "type": "slip"},
            {"faces": ["kmin"], "block": 1, "type": "wall"},
            {"faces": ["imin", "imax", "jmin", "jmax", "kmax"], "type": "farfield"},
        ],
        "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "parallel": {"devices": ["cpu"], "partitions": partitions},
    }, case_dir=tmp_path))


@pytest.mark.parametrize("precision, offset, tol", [
    ("float64", (0.0, 0.0, 0.0), 1e-13),
    ("float32", (30.0, -20.0, 30.0), 1.2e-7),  # float32 storage only; 30 units out, first cell 1e-6
])
def test_flat_plate_analytic(tmp_path, precision, offset, tol):
    s = _plate_case(tmp_path, precision, offset)
    wd = s.wall_distance()
    assert len(wd.surface.tris) == 2 * 10 * 1
    dmin = np.inf
    for sb in s.subs:
        xc = _centroids64(sb, s.ng) - np.asarray(offset)
        want = np.where(xc[..., 0] >= 0, xc[..., 2], np.hypot(xc[..., 0], xc[..., 2]))
        got = _interior(s, sb.id, wd.d[sb.id].numpy()).astype(np.float64)
        dmin = min(dmin, want.min())
        assert np.abs(got / want - 1).max() < tol
        assert np.all(_interior(s, sb.id, wd.face[sb.id].numpy()) >= 0)
    assert dmin < 1e-6


def test_faceted_cylinder_analytic(tmp_path):
    """Two half O-grid blocks around a cylinder of radius 1: the wall is the polygon through the
    nodes, so a centroid at mean radius rc (mid-angle) is (rc_nodes - 1) cos(dtheta/2) away."""
    nt, r = 17, 1.0 + np.concatenate([[0.0], 1e-3 * np.cumsum(1.3 ** np.arange(18))])
    dth = np.pi / (nt - 1)
    blocks = []
    for t0 in (0.0, np.pi):
        th = t0 + dth * np.arange(nt)
        T, R, Z = np.meshgrid(th, r, np.array([0.0, 0.2]), indexing="ij")
        blocks.append(np.stack([R * np.cos(T), R * np.sin(T), Z], -1))
    write_plot3d(tmp_path / "cyl.p3d", blocks)
    s = Solver(config_from_dict({
        "grid": {"file": "cyl.p3d"}, "flow": {"mach": 0.2, "reynolds": 1e5, "temperature": 300.0},
        "boundaries": [{"faces": ["jmin"], "type": "wall"}, {"faces": ["jmax"], "type": "farfield"},
                       {"faces": ["kmin", "kmax"], "type": "slip"}],
        "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "parallel": {"devices": ["cpu"], "partitions": 3}}, case_dir=tmp_path))
    wd = s.wall_distance()
    rc = 0.5 * (r[:-1] + r[1:])
    for sb in s.subs:
        got = _interior(s, sb.id, wd.d[sb.id].numpy())
        j0, j1 = sb.lo[1], sb.hi[1]
        want = ((rc[j0:j1] - 1.0) * np.cos(dth / 2))[None, :, None]
        assert np.abs(got / want - 1).max() < 1e-12


@pytest.mark.parametrize("partitions", [1, 3])
def test_brute_force_curvilinear_multiblock(tmp_path, partitions):
    s = _two_block_case(tmp_path, partitions)
    wd = s.wall_distance()
    V, T = wd.surface.verts, wd.surface.tris
    assert len(T) > 0
    for sb in s.subs:
        xc = _centroids64(sb, s.ng)
        want = brute_force_distance(xc.reshape(-1, 3), V, T).reshape(xc.shape[:3])
        got = _interior(s, sb.id, wd.d[sb.id].numpy())
        assert np.abs(got - want).max() <= 1e-13 * max(1.0, want.max())


def test_partition_independent(tmp_path):
    out = []
    for n in (1, 4):
        s = _plate_case(tmp_path, "float64", partitions=n)
        wd = s.wall_distance()
        parent = [np.zeros(tuple(m - 1 for m in x.shape[:3])) for x in s.parent_blocks]
        for sb in s.subs:
            lo, hi = sb.lo, sb.hi
            parent[sb.parent][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = _interior(s, sb.id, wd.d[sb.id].numpy())
        out.append(parent)
    assert all(np.array_equal(a, b) for a, b in zip(*out))


def test_no_walls(tmp_path):
    x = np.stack(np.meshgrid(*(np.linspace(0, 1, 4),) * 3, indexing="ij"), -1)
    write_plot3d(tmp_path / "box.p3d", [x])
    s = Solver(config_from_dict({
        "grid": {"file": "box.p3d"}, "flow": {"mach": 0.2, "reynolds": 1e5, "temperature": 300.0},
        "boundaries": [{"faces": ["imin", "imax", "jmin", "jmax", "kmin", "kmax"], "type": "farfield"}],
        "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "parallel": {"devices": ["cpu"], "partitions": 1}}, case_dir=tmp_path))
    wd = s.wall_distance()
    assert np.all(wd.d[0].numpy() == NO_WALL_DISTANCE) and np.all(wd.face[0].numpy() == -1)


def test_collapsed_wall_edge_dropped():
    """A wall face whose first row collapses to a point (wing tip, fuselage nose) yields no
    zero-area triangles."""
    th = np.linspace(0, np.pi / 2, 5)
    rr = np.linspace(0.0, 1.0, 4)
    T, R = np.meshgrid(rr, th, indexing="ij")
    face = np.stack([T * np.cos(R), T * np.sin(R), np.zeros_like(T)], -1)  # row 0 collapsed at the origin
    x = np.stack([face, face + [0, 0, 1.0]], axis=2)  # (4, 5, 2, 3); kmin is the wall
    surf = wall_surface([x], [(0, 4)])
    assert len(surf.tris) == 2 * 3 * 4 - 4
    a, b, c = (surf.verts[surf.tris[:, i]] for i in range(3))
    assert np.linalg.norm(np.cross(b - a, c - a), axis=1).min() > 0


def test_output_field(tmp_path):
    s = _plate_case(tmp_path, "float64")
    s.cfg.output.fields = ["rho", "wall_distance"]
    s.cfg.output.dir = str(tmp_path / "out")
    import pyvista as pv

    path = s.write_output()
    mb = pv.read(path)
    assert "wall_distance" in mb[0].cell_data and mb[1].cell_data["wall_distance"].min() > 0


def test_tmr_flatplate():
    if not (TMR / "2DZP" / "flatplate_clust2_4levelsdown_35x25.p3dfmt").exists():
        pytest.skip("TMR grid files not present")
    s = Solver(load_config(TMR / "2DZP" / "case.yaml"))
    wd = s.wall_distance()
    for sb in s.subs:
        xc = _centroids64(sb, s.ng)
        want = np.where(xc[..., 0] >= 0, xc[..., 2], np.hypot(xc[..., 0], xc[..., 2]))
        got = _interior(s, sb.id, wd.d[sb.id].numpy())
        assert np.abs(got / want - 1).max() < 1e-12


def test_ridge_near_ties_need_float64_recheck():
    """Points 1e-7..1e-5 above a sharp ridge 30 units from the surface centre: the float32 BVH
    often picks the wrong side of the ridge; the float64 recheck of the candidates in the box
    d + 4 delta recovers the float64 distance (limited only by float64 rounding at |x| ~ 30)."""
    import warp as wp

    from wmles_wrap.grid.wall_distance import EPS32, WallSurface, _wd_kernels
    from wmles_wrap.precision import get_precision

    L = 30.0
    V = np.array([[L - 5, -1, 0.9], [L - 5, 1, 0.9], [L, -1, 0], [L, 1, 0], [L + 5, -1, 0.9], [L + 5, 1, 0.9],
                  [-L, -1, 0], [-L, 1, 0], [-L + 1, -1, 0], [-L + 1, 1, 0]], dtype=np.float64)
    T = np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3], [6, 8, 9], [6, 9, 7]], dtype=np.int32)
    surf = WallSurface(V, T, 0.5 * (V.min(0) + V.max(0)))
    rng = np.random.default_rng(0)
    n = 4000
    h, lat = 10 ** rng.uniform(-7, -5, n), rng.uniform(-3e-6, 3e-6, n)
    nl, nr = np.array([0.9, 0, 5.0]) / np.hypot(0.9, 5.0), np.array([-0.9, 0, 5.0]) / np.hypot(0.9, 5.0)
    P = np.stack([L + lat, rng.uniform(-0.9, 0.9, n), 0.18 * np.abs(lat)], -1)
    P += np.where((lat > 0)[:, None], nr, nl) * h[:, None]
    want = brute_force_distance(P, V, T)

    _, k_select, k_distance = _wd_kernels(get_precision("float64"))
    pts = wp.array((V - surf.origin).astype(np.float32), dtype=wp.vec3f)
    idx = wp.array(T.ravel(), dtype=wp.int32)
    mesh = wp.Mesh(points=pts, indices=idx)
    v64, t3 = wp.array(V, dtype=wp.vec3d), wp.array(T, dtype=wp.vec3i)
    xc = wp.array(P.reshape(n, 1, 1, 3), dtype=wp.vec3d)
    err = {}
    for cap in (0, 256):
        face = wp.zeros((n, 1, 1), dtype=wp.int32)
        d = wp.zeros((n, 1, 1), dtype=wp.float64)
        wp.launch(k_select, dim=(n, 1, 1), inputs=[mesh.id, xc, v64, t3, wp.vec3d(*surf.origin), float(surf.radius),
                                                   EPS32, cap], outputs=[face])
        wp.launch(k_distance, dim=(n, 1, 1), inputs=[xc, face, v64, t3, wp.float64(1e30)], outputs=[d])
        err[cap] = np.abs(d.numpy().ravel() / want - 1).max()
    assert err[0] > 1e-2  # float32 selection alone
    assert err[256] < 1e-7
