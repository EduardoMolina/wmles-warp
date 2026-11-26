"""RANS stage R2: Roe-MUSCL steady residual and the negative Spalart-Allmaras model (numerics/sa.py).
Manufactured-solution orders, laminar Blasius and turbulent flat plate (NASA TMR 2DZP grids,
skipped when the git-ignored grid files are missing), restart and partitioning."""

import sys
from pathlib import Path

import numpy as np
import pytest

from wmles_wrap.config import ConfigError, config_from_dict, load_config
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.post.surface import wall_data
from wmles_wrap.solver import Solver

ROOT = Path(__file__).resolve().parents[1]
TMR = ROOT / "cases" / "tmr" / "2DZP"
sys.path.insert(0, str(ROOT / "verification"))


def _tmr_plate(tmp_path, grid, reynolds, turbulence, partitions=1, rans=None):
    if not (TMR / f"{grid}.p3dfmt").exists():
        pytest.skip("TMR grid files not present")
    return Solver(config_from_dict({
        "grid": {"file": f"{grid}.p3dfmt", "nmf": f"{grid}.nmf"},
        "flow": {"mach": 0.2, "reynolds": reynolds, "temperature": 300.0},
        "sgs": {"model": "none"}, "wall_model": {"type": "none"}, "numerics": {"flux": "roe_muscl"},
        "solver": {"mode": "rans"},
        "rans": {"turbulence": turbulence, "max_iterations": 800, "tolerance": 1e-10, **(rans or {})},
        "boundaries": [{"patches": ["viscous_solid"], "type": "wall"},
                       {"patches": ["symmetry_y_strong", "symmetry_z_strong"], "type": "slip"},
                       {"patches": ["subsonic_inflow_pt"], "type": "inflow_total"},
                       {"patches": ["back_pressure"], "type": "outflow"},
                       {"patches": ["farfield_riem"], "type": "farfield"}],
        "run": {"log_every": 10**6}, "output": {"dir": str(tmp_path / "out")},
        "parallel": {"devices": ["cpu"], "partitions": partitions}}, case_dir=TMR))


def _wall_cf(s, b=1):
    """Skin friction (two-cell wall derivative) and face-centroid x on the plate (kmin of block b)."""
    wd = wall_data(s)[(b, 4)]
    return wd.xf[0, :, 0], wd.cf[0]


def test_sa_manufactured_solution_orders():
    """Mean flow second order (Roe-MUSCL), nut first order (first-order upwind convection)."""
    from sa_mms import orders, run_sa_mms

    grids = [6, 12]
    runs = [run_sa_mms(n) for n in grids]
    assert all(r[2] for r in runs)  # converged to the tolerance
    o = orders(grids, [r[0] for r in runs])
    for v in ("rho", "u", "v", "w", "p"):
        assert o[v][0] > 1.7, (v, o[v])
    assert 0.85 < o["nut"][0] < 1.3


def test_roe_muscl_freestream_preservation(tmp_path):
    from helpers import _perturbed_box

    write_plot3d(tmp_path / "g.p3d", [_perturbed_box(9, 7, 6, amp=0.05)])
    s = Solver(config_from_dict({
        "grid": {"file": "g.p3d"}, "flow": {"mach": 0.3, "reynolds": 1e5, "temperature": 300.0, "alpha": 10.0},
        "numerics": {"flux": "roe_muscl"}, "sgs": {"model": "none"}, "wall_model": {"type": "none"},
        "boundaries": [{"faces": ["imin", "imax", "jmin", "jmax", "kmin", "kmax"], "type": "farfield"}],
        "parallel": {"devices": ["cpu"], "partitions": 2}}, case_dir=tmp_path))
    s.integrator.rhs(0.0, 0)
    assert max(np.abs(d.R.numpy()).max() for d in s.integrator.data.values()) < 1e-12


def test_blasius(tmp_path):
    s = _tmr_plate(tmp_path, "flatplate_clust2_2levelsdown_137x97", 1e5, "none")
    assert s.run_steady() and s.step < 400
    x, cf = _wall_cf(s)
    sel = (x > 0.2) & (x < 1.5)  # away from the plate end at the outflow (x = 2)
    ratio = cf[sel] / (0.664 / np.sqrt(1e5 * x[sel]))
    assert np.abs(ratio - 1).max() < 0.02


def test_sa_flat_plate(tmp_path):
    """TMR 2DZP, 35x25: converges to machine zero; C_f(0.97) between the CFL3D and FUN3D values
    on the same grid (2.7957e-3, 2.6674e-3); the converged value is a regression reference (TMR
    inflow_total BC, two-cell wall derivative)."""
    s = _tmr_plate(tmp_path, "flatplate_clust2_4levelsdown_35x25", 5e6, "sa")
    assert s.run_steady() and s.step < 700
    x, cf = _wall_cf(s)
    c97 = np.interp(0.970084071, x, cf)
    assert 2.6674e-3 < c97 < 2.7957e-3
    assert c97 == pytest.approx(2.766426e-3, rel=1e-5)
    nut = s.turbulence.parent_nut()[1] / s.gas.mu(np.float64(1.0))
    assert 300 < nut.max() < 500  # nut / nu_inf in the boundary layer


def test_sa_restart_and_partitioning(tmp_path):
    a = _tmr_plate(tmp_path / "a", "flatplate_clust2_4levelsdown_35x25", 5e6, "sa", partitions=1,
                   rans={"max_iterations": 30})
    a.run_steady()
    b = _tmr_plate(tmp_path / "b", "flatplate_clust2_4levelsdown_35x25", 5e6, "sa", partitions=3,
                   rans={"max_iterations": 30})
    b.run_steady()
    for x, y in zip(a.parent_states(), b.parent_states()):
        assert np.array_equal(x, y)
    for x, y in zip(a.turbulence.parent_nut(), b.turbulence.parent_nut()):
        assert np.array_equal(x, y)
    # restart into a different partitioning keeps the state and nut
    c = _tmr_plate(tmp_path / "c", "flatplate_clust2_4levelsdown_35x25", 5e6, "sa", partitions=2)
    c.read_restart(tmp_path / "a" / "out" / "restart_final.npz")
    for x, y in zip(a.turbulence.parent_nut(), c.turbulence.parent_nut()):
        assert np.array_equal(x, y)
    for x, y in zip(a.parent_states(), c.parent_states()):
        assert np.array_equal(x, y)


def test_sa_config_validation():
    base = {"grid": {"file": "g.p3d"}, "flow": {"mach": 0.2, "reynolds": 1e4, "temperature": 300.0}}
    with pytest.raises(ConfigError, match="rans.turbulence"):
        config_from_dict({**base, "rans": {"turbulence": "sst"}})
    with pytest.raises(ConfigError, match="numerics.flux"):
        config_from_dict({**base, "numerics": {"flux": "ausm"}})
    with pytest.raises(ConfigError, match="solver.mode: rans"):
        config_from_dict({**base, "rans": {"turbulence": "sa"}})
