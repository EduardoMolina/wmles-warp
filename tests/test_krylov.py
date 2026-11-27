"""RANS stage R4: FGMRES on the pseudo-time system (numerics/krylov.py). Exact solve of the line
operator system, the Jacobian-free product against central differences of the residual, inexact
Newton convergence, coupled SA, and bitwise independence of the partitioning."""

import numpy as np
import pytest
import warp as wp

from test_implicit import _plate
from test_sa import _tmr_plate, _wall_cf
from wmles_wrap.config import ConfigError, config_from_dict

FG = {"linear_solver": "fgmres"}


def _weighted(imp, a):
    """Interior array (n..., 5) scaled by 1 / (V/dtau ref), as the Krylov inner product."""
    dts = imp.s.interior(imp.lines[0].dts.numpy()).astype(np.float64)
    return a / dts[..., None] / imp.ref


def test_operator_fgmres_solves_the_operator_system(tmp_path):
    """With enough vectors, FGMRES on the approximate operator reaches the requested drop of the
    true linear residual b - A x (the Arnoldi estimate is not trusted)."""
    s = _plate(tmp_path, rans={**FG, "krylov_matvec": "operator", "krylov_restart": 60, "krylov_tolerance": 1e-9,
                               "solve_precision": "float64", "max_iterations": 15})
    s.run_steady()  # a developed, non-uniform state
    imp, kr = s.implicit, s.implicit.krylov
    s.integrator.rhs(0.0, 0)
    imp.assemble(500.0)
    kr.solve()
    assert kr.drop <= 1e-9 and kr.iterations < 60
    x = imp.lines[0].dq.numpy().copy()
    imp.assemble(500.0)  # same state: same blocks and right-hand side
    b = s.interior(imp.lines[0].rhs.numpy()).astype(np.float64)
    wp.copy(kr.ws[0]["Z"][0], wp.array(x, dtype=imp.mp.solve.vec5, device="cpu"))
    kr.matvec(0)
    r = b - s.interior(kr.ws[0]["w"].numpy())
    assert np.linalg.norm(_weighted(imp, r)) <= 2e-9 * np.linalg.norm(_weighted(imp, b))


def test_jfnk_product_is_the_residual_linearization(tmp_path):
    """J z = V/dtau z - dR/dQ z against a central difference of the residual (float64)."""
    s = _plate(tmp_path, rans={**FG, "solve_precision": "float64", "max_iterations": 8})
    s.run_steady()
    imp, kr = s.implicit, s.implicit.krylov
    I, ng = s.integrator, s.ng
    I.rhs(0.0, 0)
    imp.assemble(50.0)
    U0 = s.U[0].numpy().copy()
    for sid, ws in kr.ws.items():  # z = 0 on the other sub-block (the plate has two)
        wp.copy(ws["U0"], s.U[sid])
        wp.copy(ws["R0"], I.data[sid].R)
        ws["Z"].zero_()
    w = kr.ws[0]
    z = np.zeros(U0.shape)
    rng = np.random.default_rng(3)
    z[ng:-ng, ng:-ng, ng:-ng] = rng.standard_normal(s.interior(U0).shape) * imp.ref
    wp.copy(w["Z"][0], wp.array(z, dtype=imp.mp.solve.vec5, device="cpu"))
    kr.matvec(0)
    Jz = s.interior(w["w"].numpy())
    h = 1e-6
    R = []
    for sign in (1, -1):
        s.set_state(0, U0 + sign * h * z)
        I.rhs(0.0, 0)
        R.append(I.data[0].R.numpy().copy())
    s.set_state(0, U0)
    dts = s.interior(imp.lines[0].dts.numpy())
    ref = dts[..., None] * s.interior(z) - (R[0] - R[1]) / (2 * h)
    assert np.linalg.norm(_weighted(imp, Jz - ref)) <= 1e-6 * np.linalg.norm(_weighted(imp, ref))


def test_jfnk_inexact_newton_convergence(tmp_path):
    """Laminar plate: the line-relaxation defect correction needs 60 iterations; JFNK converges
    as inexact Newton once the CFL is large, to the same discrete solution."""
    ref = _plate(tmp_path / "ref")
    assert ref.run_steady()
    s = _plate(tmp_path / "jfnk", rans=FG)
    assert s.run_steady()
    h = s.implicit.history
    assert len(h) < 20 and h[-1][2] <= 1e-11 * h[0][2]
    assert all(r[5] <= s.cfg.rans.krylov_restart for r in h)  # Krylov iterations in the history
    for a, b in zip(ref.parent_states(), s.parent_states()):
        assert np.abs(a - b).max() < 1e-8 * np.abs(a).max()


@pytest.mark.parametrize("matvec", ["jfnk", "operator"])
def test_partition_independent(tmp_path, matvec):
    states = []
    for n in (1, 3):
        s = _plate(tmp_path / f"p{n}", partitions=n, rans={**FG, "krylov_matvec": matvec, "max_iterations": 12})
        s.run_steady()
        states.append(s.parent_states())
    assert all(np.array_equal(a, b) for a, b in zip(*states))


def test_coupled_sa_flat_plate(tmp_path):
    """TMR 2DZP 35x25 with SA: coupled (Q, nut) JFNK converges ~6x faster than the segregated
    defect correction (526 iterations) to the same C_f; bitwise independent of the partitioning."""
    grid = "flatplate_clust2_4levelsdown_35x25"
    s = _tmr_plate(tmp_path / "a", grid, 5e6, "sa", rans=FG)
    assert s.run_steady() and s.step < 120
    x, cf = _wall_cf(s)
    assert np.interp(0.970084071, x, cf) == pytest.approx(2.771848e-3, rel=1e-5)
    runs = []
    for n in (1, 3):
        r = _tmr_plate(tmp_path / f"p{n}", grid, 5e6, "sa", partitions=n, rans={**FG, "max_iterations": 25})
        r.run_steady()
        runs.append(r)
    for x0, x1 in zip(runs[0].parent_states(), runs[1].parent_states()):
        assert np.array_equal(x0, x1)
    for x0, x1 in zip(runs[0].turbulence.parent_nut(), runs[1].turbulence.parent_nut()):
        assert np.array_equal(x0, x1)


def test_krylov_config_validation():
    base = {"grid": {"file": "g.p3d"}, "flow": {"mach": 0.2, "reynolds": 1e4, "temperature": 300.0},
            "solver": {"mode": "rans"}, "sgs": {"model": "none"}, "wall_model": {"type": "none"}}
    for rans, msg in [({"linear_solver": "gmres"}, "rans.linear_solver"), ({"krylov_matvec": "exact"}, "krylov_matvec"),
                      ({"krylov_restart": 0}, "krylov_restart"), ({"krylov_tolerance": 1.0}, "krylov_tolerance"),
                      ({"fd_epsilon": 0.0}, "fd_epsilon")]:
        with pytest.raises(ConfigError, match=msg):
            config_from_dict({**base, "rans": rans})
