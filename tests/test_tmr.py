"""RANS stage R3: NASA TMR verification cases on their coarsest grids (skipped when the git-ignored
grid files are missing). The full grid studies are verification/sa_bump.py and sa_jet.py; these
tests pin the converged coarse-grid values (regression) and check them against CFL3D/FUN3D on the
same grid where the codes agree."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))


def _need(case, name):
    if not (ROOT / "cases" / "tmr" / case / name).exists():
        pytest.skip("TMR grid files not present")


def test_coflowing_jet_coarse():
    """2DCJ, 1280 cells: two inflow_total streams, zero-thickness plate, 4 blocks after patch
    splitting. Centreline u within 0.0015 a_ref of CFL3D on the same grid."""
    _need("2DCJ", "shear_4levelsdown.p3dfmt")
    import sa_jet

    n, cd, us, _, it, ok, _ = sa_jet.run(4)
    assert ok and n == 1280 and it < 400
    for u, ref in zip(us, (0.502203, 0.407499, 0.326802)):
        assert abs(u - ref) < 1.5e-3
    assert cd == pytest.approx(7.333812e-3, rel=1e-5)


def test_bump_coarse():
    """2DB, 89x41: C_f at the three TMR stations (regression), within 1.5 % of CFL3D at x = 0.632
    and 0.75 (the coarse-grid codes differ by up to 13 % at x = 0.868)."""
    _need("2DB", "bump_4levelsdown_89x41.p3dfmt")
    import sa_bump

    n, cfs, _, it, ok, _ = sa_bump.run(4)
    assert ok and n == 3520 and it < 400
    for c, ref in zip(cfs, (5.264675e-3, 5.711130e-3, 2.680479e-3)):
        assert c == pytest.approx(ref, rel=1e-5)
    for c, ref in zip(cfs[:2], (5.33636e-3, 5.73478e-3)):  # CFL3D, same grid
        assert c == pytest.approx(ref, rel=0.015)


def test_naca0012_coarse():
    """NACA 0012, alpha = 10 deg, family II.7 (3584 cells, C-grid with the wake cut as a 1-to-1
    self-interface, alpha in the x-z plane): regression of C_L, C_D, C_m (the finer-grid comparison
    with CFL3D/FUN3D/TAU is verification/sa_naca0012.py)."""
    _need("2DNACA0012", "n0012familyII.7.p3dfmt")
    import sa_naca0012

    n, nodes, c, it, ok, _ = sa_naca0012.run(7)
    assert ok and n == 3584 and nodes == 3729 and it < 500
    assert c["CL"] == pytest.approx(1.054257, rel=1e-5)
    assert c["CD"] == pytest.approx(0.02441090, rel=1e-5)
    assert c["CMy"] == pytest.approx(0.002824214, rel=1e-4)
    assert c["CD"] == pytest.approx(c["CDp"] + c["CDv"], rel=1e-12)
