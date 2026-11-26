"""RANS stage R3: total-condition inflow (inflow_total) and back-pressure outflow, the TMR open
boundaries, and their linearization in the implicit operator; wall-surface post-processing."""

import numpy as np
import pytest

from wmles_wrap.config import config_from_dict
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.numerics.implicit import ImplicitSolver
from wmles_wrap.post.surface import wall_data
from wmles_wrap.solver import Solver
from test_implicit import _fd_line_block


def _duct(tmp_path, n=8, inflow=None, numerics=None, rans=None):
    """Periodic in y and z, inflow_total on imin, outflow on imax (flow along x)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 1, n + 1)
    write_plot3d(tmp_path / "d.p3d", [np.stack(np.meshgrid(x, np.linspace(0, 0.5, 3), np.linspace(0, 0.5, 3),
                                                           indexing="ij"), -1)])
    return Solver(config_from_dict({
        "grid": {"file": "d.p3d"}, "flow": {"mach": 0.3, "reynolds": 100.0, "temperature": 300.0},
        "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "numerics": numerics or {"flux": "roe_muscl"}, "solver": {"mode": "rans"},
        "rans": {"solve_precision": "float64", **(rans or {})},
        "boundaries": [{"faces": ["imin"], "type": "inflow_total", "params": inflow or {}},
                       {"faces": ["imax"], "type": "outflow"}],
        "periodic": [{"translation": [0, 0.5, 0]}, {"translation": [0, 0, 0.5]}],
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp_path / "out")},
        "parallel": {"devices": ["cpu"], "partitions": 1}}, case_dir=tmp_path))


def _ghost_and_first(s):
    s.integrator.update_primitives()
    s.integrator.fill_ghosts(0.0)
    W = s.integrator.data[0].W.numpy().astype(np.float64)
    ng = s.ng
    return W[ng - 1, ng + 1, ng + 1], W[ng, ng + 1, ng + 1], W[-ng, ng + 1, ng + 1]


def test_inflow_total_freestream_and_conditions(tmp_path):
    s = _duct(tmp_path)
    g = s.gas
    wg, w1, _ = _ghost_and_first(s)  # free-stream initial state: the default totals reproduce it
    assert np.allclose(wg, w1, rtol=1e-13, atol=1e-13)
    # non-uniform interior: p0, T0, direction and the outgoing invariant R+ are honoured
    s = _duct(tmp_path / "b", inflow={"p_total": 1.08, "T_total": 1.02, "direction": [1.0, 0.2, 0.0]})
    s.set_primitive_field(lambda xc: np.stack([1 + 0.05 * xc[..., 0], 0.9 + 0.1 * xc[..., 0], 0 * xc[..., 0],
                                               0.02 + 0 * xc[..., 0], g.R * (1 - 0.03 * xc[..., 0])], -1))
    wg, w1, wo = _ghost_and_first(s)
    gm, R, cp = g.gamma, g.R, g.cp
    Tb, ub = wg[4] / (wg[0] * R), wg[1:4]
    assert Tb + 0.5 * ub @ ub / cp == pytest.approx(1.02, rel=1e-12)
    assert wg[4] * (1.02 / Tb) ** (gm / (gm - 1)) == pytest.approx(1.08 * R, rel=1e-12)
    assert np.allclose(np.cross(ub, [1.0, 0.2, 0.0]), 0, atol=1e-14) and ub[0] > 0
    rp = lambda w: -w[1] + 2 * np.sqrt(gm * w[4] / w[0]) / (gm - 1)  # noqa: E731  (outward n = -x)
    assert rp(wg) == pytest.approx(rp(w1), rel=1e-12)
    # back-pressure outflow: p = p_inf, the rest from the interior
    assert wo[4] == pytest.approx(R, rel=1e-14)


@pytest.mark.parametrize("numerics", [{"sensor": "none", "dissipation": "roe"}])
def test_open_boundary_blocks_against_finite_differences(tmp_path, numerics):
    """The diagonal blocks of the cells next to the inflow_total and outflow faces include the
    linearized ghosts: they match the finite-difference Jacobian of the residual."""
    s = _duct(tmp_path, n=6, numerics=numerics, rans={"jacobian": "roe", "implicit_dissipation": 1.0})
    g = s.gas
    s.set_primitive_field(lambda xc: np.stack([1 + 0.05 * xc[..., 0], 0.9 + 0.1 * xc[..., 0], 0.1 + 0 * xc[..., 0],
                                               0 * xc[..., 0], g.R * (1 + 0.04 * xc[..., 0])], -1))
    imp = ImplicitSolver(s)
    s.integrator.rhs(0.0, 0)
    imp.assemble(1e30)
    ng, n = s.ng, 6
    for c in ((ng, ng + 1, ng + 1), (ng + n - 1, ng + 1, ng + 1)):
        fd = _fd_line_block(s, imp, c, c)
        D = imp.lines[0].D.numpy()[c]
        assert np.linalg.norm(D - fd) <= 0.06 * np.linalg.norm(fd), (c, np.linalg.norm(D - fd) / np.linalg.norm(fd))


def test_wall_data_linear_and_quadratic_profiles(tmp_path):
    """Couette-like wall data: order 2 is exact for a quadratic profile, order 1 for a linear one."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    z = np.concatenate([[0.0], 0.01 * np.cumsum(1.2 ** np.arange(8))])
    write_plot3d(tmp_path / "w.p3d", [np.stack(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 0.1, 2), z,
                                                           indexing="ij"), -1)])
    s = Solver(config_from_dict({
        "grid": {"file": "w.p3d"}, "flow": {"mach": 0.2, "reynolds": 1000.0, "temperature": 300.0},
        "gas": {"viscosity": "constant"}, "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "boundaries": [{"faces": ["kmin"], "type": "wall"}, {"faces": ["kmax"], "type": "slip"},
                       {"faces": ["imin", "imax", "jmin", "jmax"], "type": "slip"}],
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp_path / "out")},
        "parallel": {"devices": ["cpu"], "partitions": 2}}, case_dir=tmp_path))
    g = s.gas
    for prof, order in ((lambda zz: 2 * zz, 1), (lambda zz: 2 * zz - 3 * zz * zz, 2)):
        s.set_primitive_field(lambda xc: np.stack([1 + 0 * xc[..., 0], prof(xc[..., 2]), 0 * xc[..., 0],
                                                   0 * xc[..., 0], g.R * (1 + 0.1 * xc[..., 0])], -1))
        wd = wall_data(s, order=order)[(0, 4)]
        assert np.allclose(wd.cf, 2 * g.mu(1.0) / 0.5, rtol=1e-12)  # tau = mu du/dz(0) = 2 mu
        assert np.allclose(wd.cp, 0.1 * wd.xf[..., 0] * g.R / 0.5, rtol=1e-12)  # p uniform in z
        F = wd.force(g.R)
        assert F[0] == pytest.approx(2 * g.mu(1.0) * 0.1, rel=1e-12)
