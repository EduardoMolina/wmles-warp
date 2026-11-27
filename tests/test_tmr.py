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
    splitting. Centreline u within 0.001 a_ref of CFL3D on the same grid."""
    _need("2DCJ", "shear_4levelsdown.p3dfmt")
    import sa_jet

    n, cd, us, _, it, ok, _ = sa_jet.run(4)
    assert ok and n == 1280 and it < 400
    for u, ref in zip(us, (0.502203, 0.407499, 0.326802)):
        assert abs(u - ref) < 1e-3
    assert cd == pytest.approx(7.65709e-3, rel=1e-5)


def test_bump_coarse():
    """2DB, 89x41: C_f at the three TMR stations (regression; the coarse-grid codes differ by up
    to 13 % here, see verification/README.md)."""
    _need("2DB", "bump_4levelsdown_89x41.p3dfmt")
    import sa_bump

    n, cfs, _, it, ok, _ = sa_bump.run(4)
    assert ok and n == 3520 and it < 400
    for c, ref in zip(cfs, (5.05959e-3, 5.35560e-3, 2.46402e-3)):
        assert c == pytest.approx(ref, rel=1e-5)
