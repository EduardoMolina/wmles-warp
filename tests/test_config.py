from pathlib import Path

import pytest

from wmles_wrap.config import ConfigError, config_from_dict, load_config

CASE = Path(__file__).resolve().parents[1] / "cases" / "channel" / "case.yaml"


def test_load_channel_case():
    cfg = load_config(CASE)
    assert cfg.flow.mach == 0.2
    assert cfg.wall_model.type == "equilibrium_ode"
    assert cfg.grid_path.exists()
    assert cfg.r_gas == pytest.approx(1 / (1.4 * 0.04))


BASE = {"grid": {"file": "g.p3d"}, "flow": {"mach": 0.5, "reynolds": 1e5, "temperature": 300.0}}


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({**BASE, "flow": {**BASE["flow"], "mahc": 0.3}})


def test_isothermal_requires_twall():
    with pytest.raises(ConfigError, match="T_wall"):
        config_from_dict({**BASE, "boundaries": [{"faces": ["jmin"], "type": "wall", "thermal": "isothermal"}]})


def test_bad_choice():
    with pytest.raises(ConfigError, match="wall_model.type"):
        config_from_dict({**BASE, "wall_model": {"type": "magic"}})
