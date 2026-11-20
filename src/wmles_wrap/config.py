"""YAML case configuration -> validated dataclasses, plus nondimensionalization.

Nondimensionalization (all solver quantities are nondimensional):
    x* = x / L_ref,  rho* = rho / rho_ref,  u* = u / U_ref,  T* = T / T_ref,
    p* = p / (rho_ref U_ref^2),  t* = t U_ref / L_ref,  U_ref = Mach * a_ref.
Hence the equation of state is p* = rho* T* / (gamma Mach^2), and
    mu* = mu(T)/mu(T_ref) / Re,   k* = mu* / ((gamma - 1) Mach^2 Pr).
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FACE_NAMES = ("imin", "imax", "jmin", "jmax", "kmin", "kmax")


class ConfigError(ValueError):
    pass


@dataclass
class ExtrudeCfg:
    nk: int
    length: float


@dataclass
class GridCfg:
    file: str
    format: str = "plot3d"
    scale: float | None = None  # coordinates are divided by flow.ref_length unless overridden
    extrude: ExtrudeCfg | None = None


@dataclass
class FlowCfg:
    mach: float
    reynolds: float
    temperature: float  # reference temperature [K]
    ref_length: float = 1.0
    alpha: float = 0.0  # angle of attack [deg] (x-y plane)
    beta: float = 0.0  # sideslip [deg]


@dataclass
class GasCfg:
    gamma: float = 1.4
    prandtl: float = 0.72
    prandtl_t: float = 0.9
    viscosity: str = "sutherland"  # sutherland | power | constant
    sutherland_s: float = 110.4  # [K]
    power_exponent: float = 0.76


@dataclass
class NumericsCfg:
    precision: str = "float64"
    flux: str = "keep_pe"
    sensor: str = "ducros_pressure"  # ducros_pressure | ducros | none
    dissipation: str = "roe"
    cfl: float = 0.8
    rk: str = "ssprk3"
    ghost_layers: int = 2
    viscous: bool = True  # False -> Euler equations
    sensor_threshold: float = 0.0  # dissipation only where Ducros theta exceeds this
    sensor_eps: float = 1e-12  # regularization of the Ducros sensor denominator
    sensor_pressure_gain: float = 10.0  # phi ~ min(1, gain * |d2p| / sum p) in ducros_pressure
    entropy_fix: float = 0.1  # Harten entropy fix (fraction of sound speed) for Roe


@dataclass
class SgsCfg:
    model: str = "vreman"  # vreman | wale | smagorinsky | none
    c: float = 0.07


@dataclass
class ExchangeCfg:
    cell_index: int | None = 3
    distance: float | None = None


@dataclass
class WallModelCfg:
    type: str = "equilibrium_ode"  # equilibrium_ode | algebraic | none
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    n_points: int = 30
    stretching: float = 1.1
    kappa: float = 0.41
    a_plus: float = 17.0


@dataclass
class BoundaryCfg:
    type: str
    faces: list[str]
    block: int | list[int] | None = None  # None -> all blocks
    thermal: str = "adiabatic"  # adiabatic | isothermal
    T_wall: float | None = None  # nondimensional (T/T_ref)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PeriodicCfg:
    translation: list[float]
    faces: list[str] | None = None  # informative; matching is geometric


@dataclass
class ForcingCfg:
    type: str = "none"  # none | body_force | constant_mass_flux
    direction: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    value: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # body_force (per unit mass)


@dataclass
class RunCfg:
    steps: int = 0
    t_end: float | None = None
    dt: float | None = None  # fixed time step; None -> CFL-based
    log_every: int = 100
    output_every: int = 0
    stats_start: int = 0
    restart_every: int = 0
    restart_file: str | None = None


@dataclass
class ParallelCfg:
    devices: str | list[str] = "auto"
    partitions: int | str = "auto"


@dataclass
class OutputCfg:
    dir: str = "out"
    fields: list[str] = field(default_factory=lambda: ["rho", "u", "v", "w", "p", "T", "mach"])
    format: str = "vtm"


@dataclass
class CaseConfig:
    grid: GridCfg
    flow: FlowCfg
    gas: GasCfg = field(default_factory=GasCfg)
    numerics: NumericsCfg = field(default_factory=NumericsCfg)
    sgs: SgsCfg = field(default_factory=SgsCfg)
    wall_model: WallModelCfg = field(default_factory=WallModelCfg)
    boundaries: list[BoundaryCfg] = field(default_factory=list)
    periodic: list[PeriodicCfg] = field(default_factory=list)
    forcing: ForcingCfg = field(default_factory=ForcingCfg)
    run: RunCfg = field(default_factory=RunCfg)
    parallel: ParallelCfg = field(default_factory=ParallelCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    case_dir: Path = field(default_factory=Path.cwd)

    # ---- derived nondimensional constants -------------------------------------------
    @property
    def r_gas(self) -> float:
        """Nondimensional gas constant: p = rho R T."""
        return 1.0 / (self.gas.gamma * self.flow.mach**2)

    @property
    def cp(self) -> float:
        return self.gas.gamma * self.r_gas / (self.gas.gamma - 1.0)

    @property
    def sutherland_s_nd(self) -> float:
        return self.gas.sutherland_s / self.flow.temperature

    @property
    def grid_path(self) -> Path:
        p = Path(self.grid.file)
        return p if p.is_absolute() else self.case_dir / p

    @property
    def grid_scale(self) -> float:
        return self.grid.scale if self.grid.scale is not None else 1.0 / self.flow.ref_length


# ---- generic dict -> dataclass conversion ----------------------------------------------

def _convert(tp, value, path: str):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(a, value, path)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(f"{path}: value {value!r} does not match {tp}: {errors}")
    if origin is list:
        if not isinstance(value, list):
            value = [value]
        return [_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping")
        return dict(value)
    if dataclasses.is_dataclass(tp):
        return _from_dict(tp, value, path)
    if tp is Any:
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected integer, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected string, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    return value


def _from_dict(cls, data, path: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {k: _convert(hints[k], v, f"{path}.{k}" if path else k) for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{path or 'case'}: {e}") from None


def _validate(cfg: CaseConfig) -> None:
    def choice(val, allowed, name):
        if val not in allowed:
            raise ConfigError(f"{name}={val!r}; expected one of {allowed}")

    choice(cfg.grid.format, ("plot3d",), "grid.format")
    choice(cfg.numerics.precision, ("float32", "float64"), "numerics.precision")
    choice(cfg.numerics.flux, ("keep_pe",), "numerics.flux")
    choice(cfg.numerics.sensor, ("ducros_pressure", "ducros", "none"), "numerics.sensor")
    choice(cfg.numerics.dissipation, ("roe", "rusanov", "none"), "numerics.dissipation")
    choice(cfg.numerics.rk, ("ssprk3",), "numerics.rk")
    choice(cfg.sgs.model, ("vreman", "wale", "smagorinsky", "none"), "sgs.model")
    choice(cfg.wall_model.type, ("equilibrium_ode", "algebraic", "none"), "wall_model.type")
    choice(cfg.gas.viscosity, ("sutherland", "power", "constant"), "gas.viscosity")
    choice(cfg.forcing.type, ("none", "body_force", "constant_mass_flux"), "forcing.type")
    if cfg.flow.mach <= 0 or cfg.flow.reynolds <= 0 or cfg.flow.temperature <= 0:
        raise ConfigError("flow.mach, flow.reynolds and flow.temperature must be positive")
    if cfg.numerics.ghost_layers < 2:
        raise ConfigError("numerics.ghost_layers must be >= 2")
    bc_types = ("wall", "wall_modeled", "slip", "farfield", "inflow", "outflow", "extrapolate", "dirichlet")
    for i, b in enumerate(cfg.boundaries):
        choice(b.type, bc_types, f"boundaries[{i}].type")
        for f in b.faces:
            choice(f, FACE_NAMES, f"boundaries[{i}].faces")
        choice(b.thermal, ("adiabatic", "isothermal"), f"boundaries[{i}].thermal")
        if b.thermal == "isothermal" and b.T_wall is None:
            raise ConfigError(f"boundaries[{i}]: isothermal wall requires T_wall")
    for i, p in enumerate(cfg.periodic):
        if len(p.translation) != 3:
            raise ConfigError(f"periodic[{i}].translation must have 3 components")
    if cfg.grid.extrude is not None and cfg.grid.extrude.nk < 2:
        raise ConfigError("grid.extrude.nk must be >= 2")


def load_config(path: str | Path) -> CaseConfig:
    path = Path(path)
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(data, case_dir=path.resolve().parent)


def config_from_dict(data: dict, case_dir: Path | None = None) -> CaseConfig:
    if "case_dir" in data:
        raise ConfigError("'case_dir' is reserved")
    cfg = _from_dict(CaseConfig, data, "")
    if case_dir is not None:
        cfg.case_dir = Path(case_dir)
    _validate(cfg)
    return cfg
