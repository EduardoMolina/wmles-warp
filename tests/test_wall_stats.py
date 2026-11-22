"""Wall-modeled face diagnostics: force monitor, surface statistics, time-based sampling and
statistics carried through restart files."""

import numpy as np
import pytest

from wmles_wrap.config import config_from_dict
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.solver import Solver

WM = "equilibrium_ode"


def _channel(tmp_path, partitions=3, run=None, forcing=True):
    y = np.linspace(-1, 1, 13)
    x = np.stack(np.meshgrid(np.linspace(0, 2, 9), y, np.linspace(0, 1, 7), indexing="ij"), axis=-1)
    x[..., 1] += 0.03 * np.sin(np.pi * x[..., 0]) * (1 - x[..., 1] ** 2)  # curvilinear interior
    write_plot3d(tmp_path / "ch.p3d", [x])
    data = {
        "grid": {"file": "ch.p3d"},
        "flow": {"mach": 0.2, "reynolds": 1e5, "temperature": 300.0, "ref_area": 2.0},
        "wall_model": {"type": WM, "exchange": {"cell_index": 2}},
        "boundaries": [{"faces": ["jmin", "jmax"], "type": "wall_modeled"}],
        "periodic": [{"translation": [2, 0, 0]}, {"translation": [0, 0, 1]}],
        "init": {"type": "channel", "amplitude": 0.1},
        "run": run or {},
        "parallel": {"devices": ["cpu"], "partitions": partitions},
    }
    if forcing:
        data["forcing"] = {"type": "constant_mass_flux"}
    return Solver(config_from_dict(data, case_dir=tmp_path))


def _wall_samples(s):
    """Host recomputation per (sid, axis, side): first-cell p, traction, heat flux, area vector
    into the fluid."""
    out = {}
    g = s.ng
    for sid, d in s.integrator.data.items():
        prim = s.gas.primitives(s.U[sid].numpy().astype(np.float64))
        Sj = s.metrics[sid].Sj.numpy().astype(np.float64)
        for (ax, side), f in d.wm.items():
            assert ax == 1
            nc = d.sub.nc[1]
            jf, jc, sgn = (g, g, 1.0) if side == 0 else (g + nc, g + nc - 1, -1.0)
            out[(sid, ax, side)] = (prim["p"][g:-g, jc, g:-g], f.tw.numpy().astype(np.float64),
                                    f.q.numpy().astype(np.float64), sgn * Sj[g:-g, jf, g:-g])
    return out


def test_wall_force_monitor(tmp_path):
    s = _channel(tmp_path)
    s.advance(nsteps=2)
    F = np.zeros(3)
    p_inf = s.gas.R
    for p, tw, _, S in _wall_samples(s).values():
        F += (tw * np.linalg.norm(S, axis=-1)[..., None]).reshape(-1, 3).sum(0)
        F -= ((p - p_inf)[..., None] * S).reshape(-1, 3).sum(0)
    c = s.force_coefficients()
    np.testing.assert_allclose(c[:3], F, rtol=1e-12, atol=1e-14)
    # alpha = 0, ref_area = 2: CD = Fx / (0.5 * 2), CL = Fy / (0.5 * 2)
    assert c[4] == pytest.approx(F[0], rel=1e-12) and c[3] == pytest.approx(F[1], rel=1e-12)
    assert F[0] > 0  # channel walls are dragged along the flow


def test_wall_stats_and_time_based_sampling(tmp_path):
    s = _channel(tmp_path, run={"stats_start_time": 0.0, "stats_every": 2})
    dt = 1e-3
    s.cfg.run.dt = dt
    samples = []
    for _ in range(6):
        s.advance(nsteps=1)
        if s.step % 2 == 0:
            samples.append(_wall_samples(s))
    assert s.integrator.stats_count == 3 == len(samples)
    s.write_stats()
    z = np.load(tmp_path / "out" / "stats.npz")
    assert list(z["wall_fields"]) == ["p", "tw_x", "tw_y", "tw_z", "q"]
    for face, sgn in (("jmin", 1.0), ("jmax", -1.0)):
        a = z[f"wall0000_{face}"]
        assert a.shape == (8, 6, 5)
        # area vectors point into the fluid: +y on jmin, -y on jmax, |S| = dx dz
        S = z[f"wallS0000_{face}"]
        np.testing.assert_allclose(S[..., 1], sgn * 0.25 / 6, rtol=1e-12)
        np.testing.assert_allclose(z[f"wallxf0000_{face}"][..., 1], -sgn, atol=1e-14)
    # parent-face means equal the averages of the host samples, assembled over the partitions
    subs = {b.id: b for b in s.subs}
    for (sid, ax, side), _ in samples[0].items():
        b = subs[sid]
        sl = (slice(b.lo[0], b.hi[0]), slice(b.lo[2], b.hi[2]))
        mean = np.mean([np.concatenate([smp[(sid, ax, side)][0][..., None], smp[(sid, ax, side)][1],
                                        smp[(sid, ax, side)][2][..., None]], axis=-1) for smp in samples], axis=0)
        np.testing.assert_allclose(z[f"wall0000_{'jmin' if side == 0 else 'jmax'}"][sl], mean, rtol=1e-12, atol=1e-15)


def test_stats_start_time_threshold(tmp_path):
    s = _channel(tmp_path, run={"stats_start_time": 2.5e-3, "stats_every": 1})
    s.cfg.run.dt = 1e-3
    s.advance(nsteps=5)  # t = 1e-3 ... 5e-3: samples at t >= 2.5e-3 -> 3 samples
    assert s.integrator.stats_count == 3


def test_restart_carries_statistics(tmp_path):
    """Run with statistics, restart midway onto a different partitioning: the final statistics
    (cells and walls) match the uninterrupted run (unforced: partition-independent bitwise up to
    the halo/BC order, so a tight tolerance)."""
    run = {"stats_start_time": 0.0, "stats_every": 1, "dt": 1e-3}
    (tmp_path / "a").mkdir()
    a = _channel(tmp_path / "a", partitions=2, run=run, forcing=False)
    a.advance(nsteps=3)
    a.write_restart(tmp_path / "a" / "r.npz")
    a.advance(nsteps=3)
    a.write_stats()
    (tmp_path / "b").mkdir()
    b = _channel(tmp_path / "b", partitions=5, run=run, forcing=False)
    b.read_restart(tmp_path / "a" / "r.npz")
    assert b.integrator.stats_count == 3 and b.step == 3
    b.advance(nsteps=3)
    b.write_stats()
    za, zb = np.load(tmp_path / "a" / "out" / "stats.npz"), np.load(tmp_path / "b" / "out" / "stats.npz")
    assert int(za["count"]) == int(zb["count"]) == 6
    for k in ("block0000", "wall0000_jmin", "wall0000_jmax"):
        np.testing.assert_allclose(zb[k], za[k], rtol=1e-12, atol=1e-14)


def test_forces_history_restart_trim(tmp_path):
    """run() writes forces.dat every log_every steps; a restart from an earlier step drops the
    rows written after it before appending."""
    run = {"steps": 4, "log_every": 1, "dt": 1e-3, "restart_every": 2}
    s = _channel(tmp_path, run=run)
    s.run()
    rows = np.loadtxt(tmp_path / "out" / "forces.dat")
    assert list(rows[:, 0]) == [1, 2, 3, 4] and np.isfinite(rows[:, 5:]).all()
    s2 = _channel(tmp_path, run=run)
    s2.read_restart(tmp_path / "out" / "restart_00000002.npz")
    s2.cfg.run.steps = 3
    s2.run()
    rows2 = np.loadtxt(tmp_path / "out" / "forces.dat")
    assert list(rows2[:, 0]) == [1, 2, 3]
    np.testing.assert_allclose(rows2[2], rows[2], rtol=1e-10)
