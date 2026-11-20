"""Phase 1 code verification tests (CPU, float64, small grids)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import _two_block_case

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))

from couette import run_couette  # noqa: E402
from mms_order import run_mms  # noqa: E402
from shock_tube import run_shock_tube  # noqa: E402
from tgv import kinetic_energy, make_tgv  # noqa: E402


def _flat_primitives(solver):
    U = np.concatenate([x.reshape(-1, 5) for x in solver.parent_states()])
    p = solver.gas.primitives(U)
    return np.stack([p[k] for k in ("rho", "u", "v", "w", "p")], axis=-1)


@pytest.mark.parametrize("dissipation", ["none", "roe", "rusanov"])
@pytest.mark.parametrize("bc, alpha", [("farfield", 20.0), ("slip", 0.0)])
def test_freestream_preservation(tmp_path, dissipation, bc, alpha):
    """Uniform flow on a curvilinear, reoriented, partitioned multiblock grid stays uniform."""
    s = _two_block_case(
        tmp_path, 3,
        extra={
            "flow": {"alpha": alpha, "beta": 10.0},
            "numerics": {"dissipation": dissipation, "sensor": "none"},  # sensor none: dissipation always on
            "boundaries": [{"faces": ["jmin", "jmax"], "type": bc, "block": 0},
                           {"faces": ["imin", "imax"], "type": bc, "block": 1}],
        },
    )
    W0 = _flat_primitives(s)
    s.advance(nsteps=10)
    assert np.abs(_flat_primitives(s) - W0).max() < 1e-12


def _perturbed_case(tmp_path, partitions):
    s = _two_block_case(
        tmp_path, partitions,
        extra={
            "numerics": {"dissipation": "roe"},
            "boundaries": [
                {"faces": ["jmin", "jmax"], "type": "wall", "block": 0, "thermal": "isothermal", "T_wall": 1.05},
                {"faces": ["imin", "imax"], "type": "wall", "block": 1, "thermal": "isothermal", "T_wall": 1.05},
            ],
            "run": {"dt": 2e-3},
        },
    )
    R = s.gas.R

    def init(xc):
        x, y, z = xc[..., 0], xc[..., 1], xc[..., 2]
        W = np.zeros(xc.shape[:-1] + (5,))
        W[..., 0] = 1.0 + 0.05 * np.sin(np.pi * x) * np.cos(2 * np.pi * z)
        W[..., 1] = 4 * y * (1 - y) + 0.1 * np.sin(2 * np.pi * z)
        W[..., 2] = 0.05 * np.sin(np.pi * x) * np.sin(np.pi * y)
        W[..., 3] = 0.05 * np.cos(np.pi * x)
        W[..., 4] = R * (1.0 + 0.1 * np.exp(-20 * ((x - 0.7) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)))
        return W

    s.set_primitive_field(init)
    return s


def test_partition_independence_bitwise(tmp_path):
    """Full solver (walls, viscous, Roe + sensor) gives bitwise identical results for any
    decomposition and for the staged (cross-device) halo path."""
    from wmles_wrap.grid.connectivity import build_halo_maps
    from wmles_wrap.parallel.halo import HaloExchanger

    ref = None
    for partitions, staged in ((1, False), (3, False), (5, False), (5, True)):
        s = _perturbed_case(tmp_path, partitions)
        if staged:
            s.halo = HaloExchanger(s.subs, build_halo_maps(s.subs, s.interfaces, s.ng), force_staged=True)
        s.advance(nsteps=4)
        states = s.parent_states()
        assert all(np.isfinite(x).all() for x in states)
        if ref is None:
            ref = states
            continue
        for a, b in zip(ref, states):
            assert np.array_equal(a, b), (partitions, staged)


@pytest.mark.parametrize("dissipation", ["none", "roe"])
def test_mms_second_order(dissipation):
    e8 = run_mms(8, dissipation=dissipation)
    e16 = run_mms(16, dissipation=dissipation)
    for v in e8:
        order = np.log2(e8[v][0] / e16[v][0])
        assert order > 1.8, (v, order)


def test_mms_euler_second_order():
    e8 = run_mms(8, viscous=False)
    e16 = run_mms(16, viscous=False)
    for v in e8:
        assert np.log2(e8[v][0] / e16[v][0]) > 1.8, v


@pytest.mark.parametrize("case, tol", [("sod", 0.015), ("lax", 0.05), ("strong", 0.15)])
def test_shock_tubes(case, tol):
    _, prim, _, errs, _ = run_shock_tube(case, 200)
    assert np.isfinite(prim["rho"]).all() and prim["rho"].min() > 0 and prim["p"].min() > 0
    assert errs["rho"] < tol, errs


def test_couette_viscous_heating():
    r = run_couette(8)
    assert r["u_err"] < 1e-12  # linear profile is reproduced exactly
    assert r["T_err"] < 0.02  # quadratic temperature, second-order error
    assert r["p_var"] < 1e-10


def test_inviscid_tgv_conservation():
    """KEEP-PE: no kinetic-energy drift beyond the physical pressure-dilatation exchange
    (O(M^2)); mass, momentum and total energy conserved to round-off."""
    s = make_tgv(12, mach=0.1, partitions=2)

    def totals():
        t = np.zeros(5)
        for b in s.subs:
            U = s.interior(s.U[b.id].numpy())
            t += (U * s.interior(s.metrics[b.id].vol.numpy())[..., None]).sum(axis=(0, 1, 2))
        return t

    k0, c0 = kinetic_energy(s), totals()
    s.advance(t_end=3.0)
    c = totals()
    assert abs(kinetic_energy(s) / k0 - 1) < 1e-3
    assert abs(c[0] / c0[0] - 1) < 1e-12 and abs(c[4] / c0[4] - 1) < 1e-12
    assert np.abs(c[1:4]).max() < 1e-12
