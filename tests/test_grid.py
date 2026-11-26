"""Grid infrastructure tests: metrics (GCL, volumes), connectivity with arbitrary index
orientation and periodicity, ghost geometry, partitioning and halo exchange."""

import numpy as np
import pytest

from wmles_wrap.grid.metrics import compute_metrics
from wmles_wrap.precision import get_precision

from helpers import LX, LY, LZ, _field, _perturbed_box, _two_block_case  # noqa: E402


def test_unit_cube_metrics():
    prec = get_precision("float64")
    x = np.stack(np.meshgrid(*(np.linspace(0, 1, 3),) * 3, indexing="ij"), axis=-1)
    m = compute_metrics(x, prec, "cpu")
    assert np.allclose(m.vol.numpy(), 0.125)
    assert np.allclose(m.Si.numpy()[..., 0], 0.25) and np.allclose(m.Si.numpy()[..., 1:], 0)
    assert np.allclose(m.Sj.numpy()[..., 1], 0.25) and np.allclose(m.Sk.numpy()[..., 2], 0.25)


@pytest.mark.parametrize("precision, tol", [("float64", 1e-14), ("float32", 1e-6)])
def test_gcl_random_grid(precision, tol):
    rng = np.random.default_rng(3)
    x = _perturbed_box(9, 8, 7) + 0.01 * rng.standard_normal((9, 8, 7, 3))
    m = compute_metrics(x, get_precision(precision), "cpu")
    assert m.gcl_max() < tol
    assert m.vol.numpy().min() > 0


def test_float32_metrics_of_thin_cells_far_from_origin():
    """A 1e-6 first cell 30 units from the origin: float32 coordinates cannot resolve it, so the
    metrics are differenced in float64 and stored in float32 (relative error ~ float32 eps)."""
    z = 30.0 + np.concatenate([[0.0], 1e-6 * np.cumsum(1.2 ** np.arange(20))])
    x = np.linspace(30.0, 30.5, 5)
    y = np.linspace(-30.0, -29.8, 4)
    g = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    g[..., 0] += 1e-3 * np.sin(g[..., 2] * 1e4)  # non-orthogonal cells
    assert np.float32(z[1]) - np.float32(z[0]) != np.float32(z[1] - z[0])  # naive float32 fails
    m64 = compute_metrics(g, get_precision("float64"), "cpu")
    m32 = compute_metrics(g, get_precision("float32"), "cpu")
    v64, v32 = m64.vol.numpy(), m32.vol.numpy()
    assert v32.dtype == np.float32 and v64.min() > 0
    assert np.abs(v32 / v64 - 1).max() < 2e-7
    for a, b in ((m64.Si, m32.Si), (m64.Sj, m32.Sj), (m64.Sk, m32.Sk)):
        a, b = a.numpy(), b.numpy()
        scale = np.linalg.norm(a, axis=-1, keepdims=True)
        assert np.abs(b - a).max(initial=0.0, where=scale > 0) <= 2e-7 * scale.max()
        assert np.all(np.abs(b - a) <= 2e-7 * scale)
    assert np.array_equal(m32.xc.numpy(), m64.xc.numpy().astype(np.float32))
    assert m32.gcl_max() < 1e-6


@pytest.mark.parametrize("partitions", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("staged", [False, True])
def test_multiblock_halo_exchange(tmp_path, partitions, flip, staged):
    from wmles_wrap.grid.connectivity import build_halo_maps
    from wmles_wrap.parallel.halo import HaloExchanger

    solver = _two_block_case(tmp_path, partitions, flip_handedness=flip)
    if staged:
        solver.halo = HaloExchanger(solver.subs, build_halo_maps(solver.subs, solver.interfaces, solver.ng),
                                    force_staged=True)
    ng = solver.ng
    assert len(solver.subs) >= partitions
    assert any(i.periodic for i in solver.interfaces)
    total_vol = 0.0
    for s in solver.subs:
        m = solver.metrics[s.id]
        assert m.gcl_max() < 1e-14
        vol = m.vol.numpy()
        assert vol.min() > 0  # including ghost cells
        total_vol += vol[ng:-ng, ng:-ng, ng:-ng].sum()
        # fill interior with the analytic field, ghosts with garbage
        xc = m.xc.numpy()
        U = np.full(s.padded_shape_cells(ng) + (5,), -1e30)
        U[ng:-ng, ng:-ng, ng:-ng, :] = _field(xc[ng:-ng, ng:-ng, ng:-ng])[..., None]
        solver.set_state(s.id, U)
    assert total_vol == pytest.approx(LX * LY * LZ, rel=1e-12)

    solver.halo.exchange(solver.U)

    checked = 0
    for s in solver.subs:
        xc = solver.metrics[s.id].xc.numpy()
        U = solver.U[s.id].numpy()
        recv = [p for p in solver.halo.pairs if p.recv == s.id]
        for p in recv:
            # entries whose donor cell is itself a ghost (edge/corner regions beyond a
            # physical boundary) depend on BC filling; check interface data only
            donor = solver.subs[p.donor]
            didx = p.donor_idx.numpy() - ng
            interior = np.all((didx >= 0) & (didx < np.array(donor.nc)), axis=1)
            idx = p.recv_idx.numpy()[interior]
            got = U[idx[:, 0], idx[:, 1], idx[:, 2], 0]
            want = _field(xc[idx[:, 0], idx[:, 1], idx[:, 2]])
            # ghost geometry is consistent with donor geometry -> exact agreement
            assert np.allclose(got, want, rtol=0, atol=1e-12), (s.id, p.donor)
            checked += len(idx)
    assert checked > 0


def test_partition_independent_assembly(tmp_path):
    states = []
    for n in (1, 4):
        solver = _two_block_case(tmp_path, n)
        for s in solver.subs:
            xc = solver.metrics[s.id].xc.numpy()
            solver.set_state(s.id, np.repeat(_field(xc)[..., None], 5, axis=-1))
        states.append(solver.parent_states())
    for a, b in zip(*states):
        assert np.array_equal(a, b)


def test_restart_roundtrip_across_partitions(tmp_path):
    s1 = _two_block_case(tmp_path, 1)
    rng = np.random.default_rng(0)
    for s in s1.subs:
        s1.set_state(s.id, rng.random(s.padded_shape_cells(s1.ng) + (5,)))
    s1.step, s1.time = 42, 1.5
    s1.write_restart(tmp_path / "r.npz")
    s2 = _two_block_case(tmp_path, 3)
    s2.read_restart(tmp_path / "r.npz")
    assert s2.step == 42 and s2.time == 1.5
    for a, b in zip(s1.parent_states(), s2.parent_states()):
        assert np.array_equal(a, b)


def test_output_vtm(tmp_path):
    pv = pytest.importorskip("pyvista")
    solver = _two_block_case(tmp_path, 2)
    path = solver.write_output()
    mb = pv.read(path)
    assert mb.n_blocks == len(solver.subs)
    assert mb[0].n_cells == solver.subs[0].ncells
    assert np.allclose(mb[0].cell_data["rho"], 1.0)
    assert (solver.out_dir / "flow.pvd").exists()


@pytest.mark.parametrize("partitions", [1, 2, 5])
def test_all_ghosts_filled_periodic_box(tmp_path, partitions):
    """Fully periodic box: halo exchange alone must fill every ghost cell, incl. edges/corners."""
    from wmles_wrap.config import config_from_dict
    from wmles_wrap.io.plot3d import write_plot3d
    from wmles_wrap.solver import Solver

    x = np.stack(np.meshgrid(*(np.linspace(0, 1, 9),) * 3, indexing="ij"), axis=-1)
    write_plot3d(tmp_path / "box.p3d", [x])
    cfg = config_from_dict(
        {
            "grid": {"file": "box.p3d"},
            "flow": {"mach": 0.3, "reynolds": 1e4, "temperature": 300.0},
            "periodic": [{"translation": t} for t in ([1, 0, 0], [0, 1, 0], [0, 0, 1])],
            "parallel": {"devices": ["cpu"], "partitions": partitions},
        },
        case_dir=tmp_path,
    )
    solver = Solver(cfg)
    ng = solver.ng
    for s in solver.subs:
        U = np.full(s.padded_shape_cells(ng) + (5,), np.nan)
        U[ng:-ng, ng:-ng, ng:-ng] = 1.0
        solver.set_state(s.id, U)
    solver.halo.exchange(solver.U)
    for s in solver.subs:
        assert not np.isnan(solver.U[s.id].numpy()).any(), s.id
