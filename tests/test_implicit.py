"""Steady implicit solver (solver.mode: rans, numerics/implicit.py): approximate Jacobian against
finite differences, convergence to machine zero, agreement with the explicit steady state, and
independence of the partitioning, the smoother and the solve precision."""

import numpy as np
import pytest

from wmles_wrap.config import ConfigError, config_from_dict
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.numerics.implicit import ImplicitSolver
from wmles_wrap.solver import Solver


def _couette(tmp_path, ny=16, mode="rans", rans=None, numerics=None, partitions=1):
    tmp_path.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0, 1, ny + 1)
    write_plot3d(tmp_path / "c.p3d", [np.stack(np.meshgrid(np.linspace(0, 1, 3), y, np.linspace(0, 1, 3),
                                                           indexing="ij"), -1)])
    return Solver(config_from_dict({
        "grid": {"file": "c.p3d"}, "flow": {"mach": 0.8, "reynolds": 10.0, "temperature": 300.0},
        "gas": {"viscosity": "constant"}, "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "numerics": numerics or {}, "solver": {"mode": mode},
        "rans": {"implicit_dissipation": 0.0, "cfl_start": 10.0, "cfl_max": 1e3, "tolerance": 1e-12,
                 "max_iterations": 200, **(rans or {})},
        "boundaries": [{"faces": ["jmin"], "type": "wall", "thermal": "isothermal", "T_wall": 1.0},
                       {"faces": ["jmax"], "type": "wall", "thermal": "isothermal", "T_wall": 1.0,
                        "params": {"u_wall": [1.0, 0.0, 0.0]}}],
        "periodic": [{"translation": [1, 0, 0]}, {"translation": [0, 0, 1]}],
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp_path / "out")},
        "parallel": {"devices": ["cpu"], "partitions": partitions}}, case_dir=tmp_path))


def _plate(tmp_path, partitions=1, rans=None):
    """Laminar flat plate (Re = 1e4 per unit length) behind a slip run-in; first cell 1e-4."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    z = np.concatenate([[0.0], 1e-4 * np.cumsum(1.35 ** np.arange(20))])
    z = z[z <= 1.2]
    y = np.array([0.0, 0.1])
    x = np.concatenate([np.linspace(-0.4, 0.0, 6)[:-1], 2.0 * np.linspace(0.0, 1.0, 17) ** 1.4])
    write_plot3d(tmp_path / "p.p3d", [np.stack(np.meshgrid(x[:6], y, z, indexing="ij"), -1),
                                      np.stack(np.meshgrid(x[5:], y, z, indexing="ij"), -1)])
    return Solver(config_from_dict({
        "grid": {"file": "p.p3d"}, "flow": {"mach": 0.2, "reynolds": 1e4, "temperature": 300.0},
        "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "numerics": {"sensor": "none", "dissipation": "rusanov"}, "solver": {"mode": "rans"},
        "rans": {"cfl_start": 5.0, "cfl_max": 1e6, "tolerance": 1e-11, "max_iterations": 400, **(rans or {})},
        "boundaries": [{"faces": ["kmin"], "block": 0, "type": "slip"},
                       {"faces": ["kmin"], "block": 1, "type": "wall"},
                       {"faces": ["jmin", "jmax"], "type": "slip"},
                       {"faces": ["imin"], "block": 0, "type": "inflow"},
                       {"faces": ["imax"], "block": 1, "type": "outflow"},
                       {"faces": ["kmax"], "type": "farfield"}],
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp_path / "out")},
        "parallel": {"devices": ["cpu"], "partitions": partitions}}, case_dir=tmp_path))


def _fd_line_block(s, imp, cell, nb):
    """Finite-difference dRes_cell/dU_nb (Res = -R) of the actual residual."""
    I, ng = s.integrator, s.ng
    U0 = s.U[0].numpy().copy()
    s.set_state(0, U0)
    I.rhs(0.0, 0)
    R0 = I.data[0].R.numpy().copy()
    J = np.zeros((5, 5))
    for k in range(5):
        h = 1e-6 * max(1.0, abs(U0[nb][k]))
        U = U0.copy()
        U[nb + (k,)] += h
        s.set_state(0, U)
        I.rhs(0.0, 0)
        J[:, k] = -(I.data[0].R.numpy() - R0)[tuple(c - ng for c in cell)] / h
    s.set_state(0, U0)
    I.rhs(0.0, 0)
    return J


@pytest.mark.parametrize("dissipation, jacobian, tol", [("none", "rusanov", 0.08), ("roe", "roe", 0.06)])
def test_line_block_against_finite_differences(tmp_path, dissipation, jacobian, tol):
    """The stored line block (inviscid + Roe + thin-layer viscous) matches the finite-difference
    Jacobian of the residual on a sheared, heated state (KEEP-PE vs Euler flux linearization and
    the frozen Roe matrix account for the rest)."""
    s = _couette(tmp_path, ny=6, numerics={"sensor": "none", "dissipation": dissipation},
                 rans={"solve_precision": "float64", "jacobian": jacobian,
                       "implicit_dissipation": 0.0 if dissipation == "none" else 1.0})
    g = s.gas
    s.set_primitive_field(lambda xc: np.stack([1 + 0.1 * xc[..., 1], xc[..., 1], 0.3 * xc[..., 1],
                                               0 * xc[..., 1], g.R * (1 + 0.2 * xc[..., 1])], -1))
    imp = ImplicitSolver(s)
    s.integrator.rhs(0.0, 0)
    imp.assemble(1e30)
    ng = s.ng
    for j in (0, 2):
        c = (ng + 1, ng + j, ng + 1)
        fd = _fd_line_block(s, imp, c, (c[0], c[1] + 1, c[2]))
        up = imp.lines[0].Up.numpy()[c]
        assert np.linalg.norm(up - fd) <= tol * np.linalg.norm(fd)


def test_couette_machine_zero_and_explicit_steady_state(tmp_path):
    s = _couette(tmp_path / "imp")
    assert s.run_steady()
    h = s.implicit.history
    assert len(h) < 60 and h[-1][2] <= 1e-12 * h[0][2]
    U_imp = s.parent_states()[0]
    W = s.gas.primitives(U_imp[1, :, 1][None, None])
    yc = 0.5 * (np.linspace(0, 1, 17)[1:] + np.linspace(0, 1, 17)[:-1])
    assert np.abs(W["u"].ravel() - yc).max() < 1e-10  # linear profile is exact for constant mu
    assert (tmp_path / "imp" / "out" / "convergence.dat").exists()
    # the explicit solver reaches the same discrete steady state. In this closed periodic box the
    # total mass is free (every mass level is steady); the explicit solver keeps the initial mass,
    # local pseudo-time stepping only conserves sum(V/dtau drho), so compare u, T and rho/mean(rho)
    e = _couette(tmp_path / "exp", mode="les")
    e.set_primitive_field(lambda xc: np.stack([np.ones_like(xc[..., 0]), xc[..., 1], 0 * xc[..., 0],
                                               0 * xc[..., 0], e.gas.R * np.ones_like(xc[..., 0])], -1))
    e.advance(t_end=80.0)
    pe, pi = e.gas.primitives(e.parent_states()[0]), s.gas.primitives(U_imp)
    for k in ("u", "T"):
        assert np.abs(pe[k] - pi[k]).max() < 1e-9
    # the explicit run is converged to ~1e-8 here (weakly damped acoustics)
    assert np.abs(pe["rho"] / pe["rho"].mean() - pi["rho"] / pi["rho"].mean()).max() < 1e-7


def test_plate_converges(tmp_path):
    s = _plate(tmp_path)
    assert s.run_steady()
    h = s.implicit.history
    assert h[-1][2] <= 1e-11 * h[0][2] and len(h) < 300
    U = s.parent_states()
    assert all(np.isfinite(u).all() for u in U)


def test_partition_independent(tmp_path):
    states = []
    for n in (1, 3):
        s = _plate(tmp_path / f"p{n}", partitions=n, rans={"max_iterations": 25})
        s.run_steady()
        states.append(s.parent_states())
    assert all(np.array_equal(a, b) for a, b in zip(*states))


@pytest.mark.parametrize("rans", [{"smoother": "line_jacobi"}, {"solve_precision": "float64"}])
def test_converged_state_independent_of_smoother_and_solve_precision(tmp_path, rans):
    """Defect correction: the inner iteration only changes the path, not the converged state."""
    ref = _plate(tmp_path / "ref")
    assert ref.run_steady()
    alt = _plate(tmp_path / "alt", rans=rans)
    assert alt.run_steady()
    for a, b in zip(ref.parent_states(), alt.parent_states()):
        assert np.abs(a - b).max() < 1e-8 * np.abs(a).max()


def test_rans_config_validation(tmp_path):
    base = {"grid": {"file": "g.p3d"}, "flow": {"mach": 0.2, "reynolds": 1e4, "temperature": 300.0},
            "solver": {"mode": "rans"}}
    with pytest.raises(ConfigError, match="sgs.model"):
        config_from_dict({**base, "wall_model": {"type": "none"}})
    with pytest.raises(ConfigError, match="wall_model"):
        config_from_dict({**base, "sgs": {"model": "none"}})
    with pytest.raises(ConfigError, match="rans.smoother"):
        config_from_dict({**base, "sgs": {"model": "none"}, "wall_model": {"type": "none"},
                          "rans": {"smoother": "point"}})
