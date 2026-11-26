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
    format: str = "plot3d"  # plot3d (binary or formatted) | cgns (structured; HDF5 or ADF)
    # NASA neutral map file with BC patches and 1-to-1 interfaces (plot3d only). With patch
    # information (nmf or cgns), blocks are split so that every patch covers whole faces, and
    # boundaries can select faces by patch label (boundaries[].patches).
    nmf: str | None = None
    base: str | None = None  # cgns: CGNSBase_t name (default: the first base)
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
    # reference area (nondimensional, units of ref_length^2) for force coefficients in forces.dat;
    # None -> only the dimensionless forces are written
    ref_area: float | None = None


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
    flux: str = "keep_pe"  # keep_pe (LES: + sensor-weighted dissipation) | roe_muscl (steady RANS)
    muscl_eps: float = 1.0e-3  # roe_muscl: van Albada regularization, relative to free-stream scales
    entropy_fix_linear: float = 0.0  # roe_muscl: Harten fix on the entropy/shear waves (fraction of a)
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
    c: float | None = None  # model constant; None -> model default (see SGS_DEFAULT_C)

    @property
    def coefficient(self) -> float:
        return self.c if self.c is not None else SGS_DEFAULT_C.get(self.model, 0.0)


SGS_DEFAULT_C = {"vreman": 0.07, "wale": 0.5, "smagorinsky": 0.17}


@dataclass
class ExchangeCfg:
    cell_index: int | None = 3
    distance: float | None = None


@dataclass
class WallModelCfg:
    type: str = "equilibrium_ode"  # equilibrium_ode | algebraic | none
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    n_points: int = 40  # wall-model grid points (<= 64)
    stretching: float = 1.15  # geometric stretching of the wall-model grid
    iterations: int = 30  # fixed-point iterations, cold start (equilibrium model)
    iterations_warm: int = 4  # fixed-point iterations when warm-started from the previous tau_w
    update: str = "step"  # step: evaluate once per time step (first RK stage) | stage: every RK stage
    kappa: float = 0.41
    a_plus: float = 17.0


@dataclass
class BoundaryCfg:
    type: str
    faces: list[str] = field(default_factory=list)  # face names; empty -> any face
    # patch labels from the grid file (NMF patch names; CGNS BC names, BC types and families);
    # empty -> any patch. At least one of faces/patches must be given.
    patches: list[str] = field(default_factory=list)
    block: int | list[int] | None = None  # None -> all blocks (indices after patch splitting)
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
    target: float = 1.0  # constant_mass_flux: bulk momentum per unit volume along direction


@dataclass
class RunCfg:
    steps: int = 0
    t_end: float | None = None
    dt: float | None = None  # fixed time step; None -> CFL-based
    log_every: int = 100
    stats_every: int = 1
    output_every: int = 0
    stats_start: int = 0
    # start statistics at this (nondimensional) time instead of a step; takes precedence over
    # stats_start. Samples are taken on steps that are multiples of stats_every.
    stats_start_time: float | None = None
    restart_every: int = 0
    restart_keep: int = 0  # keep only the N most recent periodic restart_*.npz files (0: keep all)
    restart_file: str | None = None


@dataclass
class InitCfg:
    type: str = "freestream"  # freestream | channel
    # channel: turbulent mean profile u = (8/7) U_b (1 - |eta|)^(1/7) along forcing.direction,
    # eta = (y - center) / half_height, plus divergence-free-ish perturbations
    center: float = 0.0
    half_height: float = 1.0
    amplitude: float = 0.1
    seed: int = 0


@dataclass
class ParallelCfg:
    devices: str | list[str] = "auto"
    partitions: int | str = "auto"
    # never bisect a parent block along its line axis (wall-normal, else highest aspect ratio), so
    # implicit line solves see whole lines (RANS); off for LES
    keep_lines: bool = False


@dataclass
class SolverCfg:
    mode: str = "les"  # les: explicit SSP-RK3 | rans: steady, implicit pseudo-time (docs/plan_rans.md)


@dataclass
class RansCfg:
    """Steady implicit solver (solver.mode: rans). The residual precision is numerics.precision."""

    turbulence: str = "none"  # none: laminar | sa: negative Spalart-Allmaras (numerics/sa.py)
    nut_inf_ratio: float = 3.0  # free-stream nu tilde / nu (NASA TMR: 3)
    smoother: str = "zebra"  # zebra (red-black line Gauss-Seidel) | line_jacobi
    # linear sweeps per outer iteration: stop when max|dQ_new - dQ_old| <= sweep_tolerance max|dQ|
    # (max norms: independent of the partitioning), after min_sweeps and at most sweeps
    sweeps: int = 32
    min_sweeps: int = 4
    sweep_tolerance: float = 0.02
    solve_precision: str = "float32"  # precision of the linear relaxation (Jacobian blocks, dQ)
    cfl_start: float = 5.0
    cfl_max: float = 1.0e4
    ser_exponent: float = 1.0  # CFL = cfl_start * (res0 / res)^ser_exponent, capped by cfl_max
    max_iterations: int = 2000
    tolerance: float = 1.0e-12  # stop when the residual norm drops by this factor
    relax_limit: float = 0.5  # largest relative density/pressure change per cell and iteration
    # dissipation in the implicit operator and its scale (1 = as in a fully upwind residual); roe
    # matches a Roe residual more closely but can limit-cycle (numerics/implicit.py)
    # auto: roe when the residual's dissipation is Roe (flux roe_muscl, or keep_pe with dissipation
    # roe), else rusanov; the operator must not be less dissipative than the residual
    jacobian: str = "auto"  # auto | rusanov | roe
    implicit_dissipation: float = 1.25  # > 1: second-order residuals respond more strongly than the first-order operator
    # roe Jacobian only (the residual is not affected): Harten floors |lambda| >= fix * a of the
    # acoustic waves and of the entropy/shear waves
    jacobian_entropy_fix: float = 0.2
    jacobian_entropy_fix_linear: float = 0.05


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
    init: InitCfg = field(default_factory=InitCfg)
    parallel: ParallelCfg = field(default_factory=ParallelCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    solver: SolverCfg = field(default_factory=SolverCfg)
    rans: RansCfg = field(default_factory=RansCfg)
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
    def nmf_path(self) -> Path | None:
        if self.grid.nmf is None:
            return None
        p = Path(self.grid.nmf)
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

    choice(cfg.grid.format, ("plot3d", "cgns"), "grid.format")
    if cfg.grid.nmf is not None and cfg.grid.format != "plot3d":
        raise ConfigError("grid.nmf is only used with grid.format: plot3d")
    if cfg.grid.base is not None and cfg.grid.format != "cgns":
        raise ConfigError("grid.base is only used with grid.format: cgns")
    if cfg.grid.extrude is not None and (cfg.grid.nmf is not None or cfg.grid.format == "cgns"):
        raise ConfigError("grid.extrude cannot be combined with patch information (grid.nmf or cgns)")
    choice(cfg.numerics.precision, ("float32", "float64"), "numerics.precision")
    choice(cfg.numerics.flux, ("keep_pe", "roe_muscl"), "numerics.flux")
    if cfg.numerics.muscl_eps <= 0 or cfg.numerics.entropy_fix_linear < 0:
        raise ConfigError("numerics.muscl_eps must be > 0 and numerics.entropy_fix_linear >= 0")
    choice(cfg.numerics.sensor, ("ducros_pressure", "ducros", "none"), "numerics.sensor")
    choice(cfg.numerics.dissipation, ("roe", "rusanov", "none"), "numerics.dissipation")
    choice(cfg.numerics.rk, ("ssprk3",), "numerics.rk")
    choice(cfg.sgs.model, ("vreman", "wale", "smagorinsky", "none"), "sgs.model")
    choice(cfg.wall_model.type, ("equilibrium_ode", "algebraic", "none"), "wall_model.type")
    choice(cfg.gas.viscosity, ("sutherland", "power", "constant"), "gas.viscosity")
    choice(cfg.forcing.type, ("none", "body_force", "constant_mass_flux"), "forcing.type")
    choice(cfg.init.type, ("freestream", "channel"), "init.type")
    choice(cfg.wall_model.update, ("step", "stage"), "wall_model.update")
    if not 3 <= cfg.wall_model.n_points <= 64:
        raise ConfigError("wall_model.n_points must be in [3, 64]")
    ex = cfg.wall_model.exchange
    if (ex.distance is None or ex.distance <= 0) and (ex.cell_index is None or ex.cell_index < 1):
        raise ConfigError("wall_model.exchange needs cell_index >= 1 or distance > 0")
    if cfg.flow.mach <= 0 or cfg.flow.reynolds <= 0 or cfg.flow.temperature <= 0:
        raise ConfigError("flow.mach, flow.reynolds and flow.temperature must be positive")
    if cfg.numerics.ghost_layers < 2:
        raise ConfigError("numerics.ghost_layers must be >= 2")
    bc_types = ("wall", "wall_modeled", "slip", "farfield", "inflow", "inflow_total", "outflow", "extrapolate",
                "dirichlet")
    for i, b in enumerate(cfg.boundaries):
        choice(b.type, bc_types, f"boundaries[{i}].type")
        if not b.faces and not b.patches:
            raise ConfigError(f"boundaries[{i}]: give faces and/or patches")
        for f in b.faces:
            choice(f, FACE_NAMES, f"boundaries[{i}].faces")
        choice(b.thermal, ("adiabatic", "isothermal"), f"boundaries[{i}].thermal")
        if b.thermal == "isothermal" and b.T_wall is None:
            raise ConfigError(f"boundaries[{i}]: isothermal wall requires T_wall")
    for i, p in enumerate(cfg.periodic):
        if len(p.translation) != 3:
            raise ConfigError(f"periodic[{i}].translation must have 3 components")
    choice(cfg.solver.mode, ("les", "rans"), "solver.mode")
    choice(cfg.rans.turbulence, ("none", "sa"), "rans.turbulence")
    if cfg.rans.nut_inf_ratio < 0:
        raise ConfigError("rans.nut_inf_ratio must be >= 0")
    choice(cfg.rans.smoother, ("zebra", "line_jacobi"), "rans.smoother")
    choice(cfg.rans.solve_precision, ("float32", "float64"), "rans.solve_precision")
    choice(cfg.rans.jacobian, ("auto", "rusanov", "roe"), "rans.jacobian")
    if cfg.rans.sweeps < 1 or cfg.rans.max_iterations < 0 or not 1 <= cfg.rans.min_sweeps <= cfg.rans.sweeps:
        raise ConfigError("rans: need 1 <= min_sweeps <= sweeps and max_iterations >= 0")
    if cfg.rans.sweep_tolerance < 0:
        raise ConfigError("rans.sweep_tolerance must be >= 0")
    if not 0 < cfg.rans.cfl_start <= cfg.rans.cfl_max:
        raise ConfigError("rans: need 0 < cfl_start <= cfl_max")
    if cfg.rans.jacobian_entropy_fix < 0 or cfg.rans.jacobian_entropy_fix_linear < 0:
        raise ConfigError("rans.jacobian_entropy_fix(_linear) must be >= 0")
    if cfg.rans.implicit_dissipation < 0:
        raise ConfigError("rans.implicit_dissipation must be >= 0")
    if not 0 < cfg.rans.relax_limit < 1:
        raise ConfigError("rans.relax_limit must be in (0, 1)")
    if cfg.rans.turbulence != "none" and cfg.solver.mode != "rans":
        raise ConfigError("rans.turbulence needs solver.mode: rans")
    if cfg.solver.mode == "rans":
        if cfg.sgs.model != "none":
            raise ConfigError("solver.mode: rans needs sgs.model: none")
        if cfg.wall_model.type != "none":
            raise ConfigError("solver.mode: rans needs wall_model.type: none (wall_modeled faces are no-slip walls)")
        if cfg.forcing.type == "constant_mass_flux":
            raise ConfigError("solver.mode: rans does not support forcing.type: constant_mass_flux")
    if cfg.flow.ref_area is not None and cfg.flow.ref_area <= 0:
        raise ConfigError("flow.ref_area must be positive")
    if cfg.run.stats_every < 1:
        raise ConfigError("run.stats_every must be >= 1")
    if cfg.run.restart_keep < 0:
        raise ConfigError("run.restart_keep must be >= 0")
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
