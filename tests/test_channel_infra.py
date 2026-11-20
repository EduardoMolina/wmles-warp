"""Channel infrastructure: mass-flux control, channel initial condition, statistics."""

import sys
from pathlib import Path

import numpy as np
import pytest

from wmles_wrap.config import config_from_dict
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.solver import Solver

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))
from channel_profiles import profiles  # noqa: E402


def _channel(tmp_path, partitions=1, stats_start=0):
    y = np.tanh(1.5 * np.linspace(-1, 1, 17)) / np.tanh(1.5)
    x = np.stack(np.meshgrid(np.linspace(0, 2, 9), y, np.linspace(0, 1, 9), indexing="ij"), axis=-1)
    write_plot3d(tmp_path / "ch.p3d", [x])
    cfg = config_from_dict(
        {
            "grid": {"file": "ch.p3d"},
            "flow": {"mach": 0.3, "reynolds": 500.0, "temperature": 300.0},
            "sgs": {"model": "wale"},
            "boundaries": [{"faces": ["jmin", "jmax"], "type": "wall", "thermal": "isothermal", "T_wall": 1.0}],
            "periodic": [{"translation": [2, 0, 0]}, {"translation": [0, 0, 1]}],
            "forcing": {"type": "constant_mass_flux", "target": 1.0},
            "init": {"type": "channel", "amplitude": 0.2},
            "run": {"stats_start": stats_start, "stats_every": 2},
            "parallel": {"devices": ["cpu"], "partitions": partitions},
            "output": {"dir": "out"},
        },
        case_dir=tmp_path,
    )
    return Solver(cfg)


def test_mass_flux_control_and_stats(tmp_path):
    s = _channel(tmp_path, partitions=3, stats_start=4)
    assert s.integrator.bulk_momentum() == pytest.approx(1.0, abs=1e-13)
    s.advance(nsteps=12)
    assert s.integrator.bulk_momentum() == pytest.approx(1.0, abs=1e-12)
    f = np.array(s.integrator.forcing_history)
    assert len(f) == 12 and np.all(np.isfinite(f))
    assert s.integrator.stats_count == 5  # steps 4, 6, 8, 10, 12
    s.write_stats()
    assert (s.out_dir / "stats.npz").exists() and (s.out_dir / "stats.pvd").exists()
    prof, info = profiles(s.out_dir / "stats.npz", reynolds=500.0, mach=0.3)
    assert np.all(np.isfinite(prof["u+"])) and info["utau"] > 0
    assert len(prof["y+"]) == 8 and np.all(np.diff(prof["y+"]) > 0)


def test_channel_init_partition_independent(tmp_path):
    # equal to round-off: the bulk-momentum correction uses a global sum whose summation order
    # depends on the partitioning (forced runs are not bitwise partition-independent)
    a = _channel(tmp_path, partitions=1).parent_states()
    b = _channel(tmp_path, partitions=4).parent_states()
    assert np.allclose(a[0], b[0], rtol=1e-14, atol=1e-14)
