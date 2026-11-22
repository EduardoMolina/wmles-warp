"""Right-hand side evaluation and SSP-RK3 time integration over all sub-blocks/devices.

Per stage:
    U -> W (interior) -> halo stage 0 -> boundary conditions -> halo stages 1, 2
      -> gradients + Ducros sensor -> face fluxes (i, j, k) -> residual -> RK update
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import warp as wp

from ..grid.block import SubBlock, face_axis, face_side
from ..grid.connectivity import _ghost_slab
from ..parallel.halo import _kernels as _halo_kernels
from .kernels import (
    BC_DIRICHLET,
    BC_EXTRAPOLATE,
    BC_FARFIELD,
    BC_INFLOW,
    BC_OUTFLOW,
    BC_SLIP,
    BC_WALL_ADIABATIC,
    BC_WALL_ISOTHERMAL,
    DISS_NONE,
    DISS_ROE,
    DISS_RUSANOV,
    SGS_NONE,
    SGS_SMAGORINSKY,
    SGS_VREMAN,
    SGS_WALE,
    VISC_CONSTANT,
    VISC_POWER,
    VISC_SUTHERLAND,
    WM_ALGEBRAIC,
    WM_EQUILIBRIUM,
    WM_NONE,
    STAT_FIELDS,
    WALL_FIELDS,
    make_kernels,
    vec5d,
    vec16d,
)

# SSP-RK3 (Shu & Osher): U^{(s)} = a_s U^n + b_s (U^{(s-1)} + dt L(U^{(s-1)})), stage time c_s
SSPRK3 = ((0.0, 1.0, 0.0), (0.75, 0.25, 1.0), (1.0 / 3.0, 2.0 / 3.0, 0.5))

NO_DISSIPATION_BCS = ("wall", "wall_modeled", "slip")


def bc_code(bc, wall_model: str = "none") -> int:
    if bc.type == "wall_modeled" and wall_model != "none":
        # impermeable (slip-mirror) ghosts; the viscous wall flux comes from the wall model
        return BC_SLIP
    if bc.type in ("wall", "wall_modeled"):
        return BC_WALL_ISOTHERMAL if bc.thermal == "isothermal" else BC_WALL_ADIABATIC
    return {
        "slip": BC_SLIP,
        "farfield": BC_FARFIELD,
        "extrapolate": BC_EXTRAPOLATE,
        "inflow": BC_INFLOW,
        "outflow": BC_OUTFLOW,
        "dirichlet": BC_DIRICHLET,
    }[bc.type]


@dataclass
class _BCLaunch:
    axis: int
    side: int
    code: int
    t_wall: float
    u_wall: tuple
    p_exit: float
    dims: tuple


@dataclass
class _WMFace:
    axis: int
    side: int
    iso: int
    t_wall: float
    tw: wp.array  # (n_t1, n_t2) vec3 wall traction tau_w * t (on interior tangential cells)
    q: wp.array  # (n_t1, n_t2) heat flux k dT/dn (n into the fluid)
    ut: wp.array  # (n_t1, n_t2) friction velocity (diagnostic)
    warm: bool = False  # tw holds a previous solution -> warm start
    force: wp.array | None = None  # (n_t1, n_t2) vec3 force on the wall per face (wall_sample scratch)
    acc: wp.array | None = None  # (n_t1, n_t2) float64 sums of WALL_FIELDS (statistics)


@dataclass
class _Dirichlet:
    idx: wp.array  # padded ghost indices (vec3i) on device
    xc: np.ndarray  # (N, 3) ghost centroids (host)


@dataclass
class _SubData:
    sub: SubBlock
    W: wp.array
    U0: wp.array
    R: wp.array
    GU: wp.array
    GT: wp.array
    th: wp.array
    mut: wp.array
    F: list
    src: wp.array
    xc_interior: np.ndarray
    bcs: list = field(default_factory=list)
    nodiss: list = field(default_factory=list)  # per axis (lo, hi)
    dirichlet: list = field(default_factory=list)
    wm: dict = field(default_factory=dict)  # (axis, side) -> _WMFace


class Integrator:
    def __init__(self, solver):
        self.s = solver
        cfg = solver.cfg
        self.prec = solver.prec
        self.ng = solver.ng
        self.K = make_kernels(self.prec)
        self.params = self._make_params()
        self.source_fn: Callable | None = None  # (xc (N,3), t) -> (N,5) conservative source
        self.exact_fn: Callable | None = None  # (xc (N,3), t) -> (N,5) primitives (dirichlet)
        self.force = np.zeros(3)
        if cfg.forcing.type == "body_force":
            self.force = np.asarray(cfg.forcing.value, dtype=np.float64)
        self._dummy_wm = {}
        self.data: dict[int, _SubData] = {sb.id: self._alloc(sb) for sb in solver.subs}
        self.dtmin = {d: wp.zeros(1, dtype=self.prec.scalar, device=d) for d in solver.devices}
        # constant mass flux control
        self.mass_flux = cfg.forcing.type == "constant_mass_flux"
        d = np.asarray(cfg.forcing.direction, dtype=np.float64)
        self.force_dir = d / np.linalg.norm(d)
        self.forcing_history: list[tuple[float, float]] = []  # (t, equivalent body force per volume)
        self._vol_total = sum(float(self.s.interior(self.s.metrics[i].vol.numpy()).astype(np.float64).sum())
                              for i in self.data)
        self._msum = {d: wp.zeros(1, dtype=wp.float64, device=d) for d in solver.devices}
        # statistics
        self.stats_count = 0
        self.stats: dict[int, wp.array] = {}

    # ---- setup ----------------------------------------------------------------------------
    def _make_params(self):
        cfg, gas = self.s.cfg, self.s.gas
        sc = self.prec.scalar
        P = self.K["Params"]()
        P.gamma = sc(gas.gamma)
        P.R = sc(gas.R)
        P.cv = sc(gas.cv)
        P.cp = sc(gas.cp)
        P.prandtl = sc(gas.prandtl)
        P.prandtl_t = sc(gas.prandtl_t)
        P.visc_model = {"sutherland": VISC_SUTHERLAND, "power": VISC_POWER, "constant": VISC_CONSTANT}[gas.viscosity]
        P.s_nd = sc(gas.s_nd)
        P.power_exp = sc(gas.power_exponent)
        P.inv_re = sc(1.0 / gas.reynolds)
        rho, u, v, w, T = gas.freestream(cfg.flow.alpha, cfg.flow.beta)
        P.w_inf = self.prec.vec5(rho, u, v, w, rho * gas.R * T)
        P.sensor_eps = sc(cfg.numerics.sensor_eps)
        P.sensor_threshold = sc(cfg.numerics.sensor_threshold)
        P.entropy_fix = sc(cfg.numerics.entropy_fix)
        P.diss_type = {"none": DISS_NONE, "rusanov": DISS_RUSANOV, "roe": DISS_ROE}[cfg.numerics.dissipation]
        P.use_sensor = {"none": 0, "ducros": 1, "ducros_pressure": 2}[cfg.numerics.sensor]
        P.pressure_gain = sc(cfg.numerics.sensor_pressure_gain)
        P.viscous = 1 if cfg.numerics.viscous else 0
        P.sgs_model = {"none": SGS_NONE, "vreman": SGS_VREMAN, "wale": SGS_WALE,
                       "smagorinsky": SGS_SMAGORINSKY}[cfg.sgs.model] if cfg.numerics.viscous else SGS_NONE
        P.sgs_c = sc(cfg.sgs.coefficient)
        wm = cfg.wall_model
        P.wm_model = {"none": WM_NONE, "equilibrium_ode": WM_EQUILIBRIUM, "algebraic": WM_ALGEBRAIC}[wm.type]
        P.wm_kappa = sc(wm.kappa)
        P.wm_aplus = sc(wm.a_plus)
        P.wm_n = wm.n_points
        P.wm_r = sc(wm.stretching)
        P.wm_iters = wm.iterations
        P.wm_iters_warm = wm.iterations_warm
        return P

    def _alloc(self, sb: SubBlock) -> _SubData:
        ng, prec, dev = self.ng, self.prec, sb.device
        pad = sb.padded_shape_cells(ng)
        nc = sb.nc
        z = lambda shape, dt: wp.zeros(shape, dtype=dt, device=dev)  # noqa: E731
        F = [
            z((nc[0] + 1, nc[1], nc[2]), prec.vec5),
            z((nc[0], nc[1] + 1, nc[2]), prec.vec5),
            z((nc[0], nc[1], nc[2] + 1), prec.vec5),
        ]
        xc = self.s.metrics[sb.id].xc.numpy().astype(np.float64)
        d = _SubData(
            sub=sb,
            W=z(pad, prec.vec5),
            U0=z(pad, prec.vec5),
            R=z(nc, prec.vec5),
            GU=z(pad, self.K["mat33"]),
            GT=z(pad, prec.vec3),
            th=z(pad, prec.scalar),
            mut=z(pad, prec.scalar),
            F=F,
            src=z((1, 1, 1), prec.vec5),
            xc_interior=xc[ng:-ng, ng:-ng, ng:-ng],
        )
        d.nodiss = [[0, 0], [0, 0], [0, 0]]
        for f, bc in sb.face_bc.items():
            ax, side = face_axis(f), face_side(f)
            code = bc_code(bc, self.s.cfg.wall_model.type)
            if bc.type == "wall_modeled" and self.s.cfg.wall_model.type != "none":
                tdims = tuple(nc[a] for a in range(3) if a != ax)
                d.wm[(ax, side)] = _WMFace(
                    ax, side, 1 if bc.thermal == "isothermal" else 0, float(bc.T_wall or 1.0),
                    z(tdims, prec.vec3), z(tdims, prec.scalar), z(tdims, prec.scalar))
            tdims = tuple(pad[a] for a in range(3) if a != ax)
            u_wall = tuple(float(x) for x in bc.params.get("u_wall", (0.0, 0.0, 0.0)))
            p_exit = float(bc.params.get("p_exit", self.s.gas.R))  # default: free-stream p (rho=T=1)
            d.bcs.append(_BCLaunch(ax, side, code, float(bc.T_wall or 1.0), u_wall, p_exit, tdims))
            if bc.type in NO_DISSIPATION_BCS:
                d.nodiss[ax][side] = 1
            if code == BC_DIRICHLET:
                idx = _ghost_slab(nc, f, ng) + ng
                d.dirichlet.append(
                    _Dirichlet(
                        idx=wp.array(idx.astype(np.int32), dtype=wp.vec3i, device=dev),
                        xc=xc[idx[:, 0], idx[:, 1], idx[:, 2]],
                    )
                )
        # apply BCs in a fixed axis order for reproducible edge/corner values
        d.bcs.sort(key=lambda b: (b.axis, b.side))
        return d

    # ---- RHS ------------------------------------------------------------------------------
    def update_primitives(self) -> None:
        K = self.K
        for sid, d in self.data.items():
            wp.launch(K["prim"], dim=d.sub.nc, inputs=[self.s.U[sid], self.ng, self.params], outputs=[d.W],
                      device=d.sub.device)

    def fill_ghosts(self, t: float) -> None:
        K, ng = self.K, self.ng
        W = {sid: d.W for sid, d in self.data.items()}
        self.s.halo.exchange(W, stages=(0,))
        for sid, d in self.data.items():
            m = self.s.metrics[sid]
            S = (m.Si, m.Sj, m.Sk)
            for b in d.bcs:
                if b.code == BC_DIRICHLET:
                    continue
                wp.launch(
                    K["bc"],
                    dim=b.dims,
                    inputs=[d.W, S[b.axis], b.axis, b.side, ng, d.sub.nc[b.axis], b.code,
                            self.prec.scalar(b.t_wall), self.prec.vec3(*b.u_wall), self.prec.scalar(b.p_exit),
                            self.params],
                    device=d.sub.device,
                )
            for dr in d.dirichlet:
                if self.exact_fn is None:
                    raise RuntimeError("dirichlet boundary requires integrator.exact_fn")
                vals = wp.array(np.asarray(self.exact_fn(dr.xc, t)).astype(self.prec.np_dtype),
                                dtype=self.prec.vec5, device=d.sub.device)
                scatter = _halo_kernels(self.prec.vec5)[1]
                wp.launch(scatter, dim=len(dr.xc), inputs=[vals, dr.idx], outputs=[d.W], device=d.sub.device)
        self.s.halo.exchange(W, stages=(1, 2))

    def rhs(self, t: float, stage: int = 0) -> None:
        K, ng = self.K, self.ng
        update_wm = stage == 0 or self.s.cfg.wall_model.update == "stage"
        self.update_primitives()
        self.fill_ghosts(t)
        force = self.prec.vec3(*self.force)
        for sid, d in self.data.items():
            m = self.s.metrics[sid]
            dev = d.sub.device
            pad = d.sub.padded_shape_cells(ng)
            wp.launch(K["gradients"], dim=tuple(n - 2 for n in pad),
                      inputs=[d.W, m.Si, m.Sj, m.Sk, m.vol, self.params], outputs=[d.GU, d.GT, d.th], device=dev)
            if self.params.sgs_model != SGS_NONE:
                wp.launch(K["sgs"], dim=tuple(n - 2 for n in pad), inputs=[d.W, d.GU, m.vol, self.params],
                          outputs=[d.mut], device=dev)
            if update_wm:
                self._wall_models(d, m)
            S = (m.Si, m.Sj, m.Sk)
            for ax in range(3):
                Fa = d.F[ax]
                lo = d.wm.get((ax, 0)) or self._dummy(dev)
                hi = d.wm.get((ax, 1)) or self._dummy(dev)
                wp.launch(
                    K["flux"],
                    dim=Fa.shape,
                    inputs=[d.W, d.GU, d.GT, d.th, d.mut, S[ax], m.xc, ax, ng, Fa.shape[ax],
                            d.nodiss[ax][0], d.nodiss[ax][1], int((ax, 0) in d.wm), int((ax, 1) in d.wm),
                            lo.tw, lo.q, hi.tw, hi.q, self.params],
                    outputs=[Fa],
                    device=dev,
                )
            has_src = 0
            if self.source_fn is not None:
                srcv = np.asarray(self.source_fn(d.xc_interior.reshape(-1, 3), t)).reshape(d.sub.nc + (5,))
                d.src = wp.array(srcv.astype(self.prec.np_dtype), dtype=self.prec.vec5, device=dev)
                has_src = 1
            wp.launch(K["residual"], dim=d.sub.nc,
                      inputs=[d.F[0], d.F[1], d.F[2], d.W, m.vol, d.src, has_src, force, ng], outputs=[d.R],
                      device=dev)

    def _dummy(self, dev):
        if dev not in self._dummy_wm:
            z = lambda dt: wp.zeros((1, 1), dtype=dt, device=dev)  # noqa: E731
            self._dummy_wm[dev] = _WMFace(0, 0, 0, 1.0, z(self.prec.vec3), z(self.prec.scalar), z(self.prec.scalar))
        return self._dummy_wm[dev]

    def _wall_models(self, d: _SubData, m) -> None:
        """Evaluate the wall model on every wall-modeled face; warm-started from the stored
        tau_w after the first call."""
        if not d.wm:
            return
        ex = self.s.cfg.wall_model.exchange
        cell_index = int(ex.cell_index or 1)
        h_dist = float(ex.distance) if ex.distance else 0.0
        for (ax, side), f in d.wm.items():
            wp.launch(
                self.K["wall_model"], dim=f.tw.shape,
                inputs=[d.W, m.xc, ax, side, self.ng, d.sub.nc[ax], cell_index, self.prec.scalar(h_dist), f.iso,
                        self.prec.scalar(f.t_wall), int(f.warm), self.params],
                outputs=[f.tw, f.q, f.ut], device=d.sub.device)
            f.warm = True

    def wall_friction_velocity(self) -> float:
        """Area-unweighted mean friction velocity over all wall-modeled faces (diagnostic)."""
        vals = [f.ut.numpy().ravel() for d in self.data.values() for f in d.wm.values()]
        return float(np.concatenate(vals).mean()) if vals else float("nan")

    # ---- time stepping --------------------------------------------------------------------
    def compute_dt(self, cfl: float) -> float:
        K, ng = self.K, self.ng
        self.update_primitives()
        for arr in self.dtmin.values():
            arr.fill_(1e30)
        for sid, d in self.data.items():
            m = self.s.metrics[sid]
            wp.launch(K["dt"], dim=d.sub.nc,
                      inputs=[d.W, d.mut, m.Si, m.Sj, m.Sk, m.vol, self.prec.scalar(cfl), ng, self.params],
                      outputs=[self.dtmin[d.sub.device]], device=d.sub.device)
        return float(min(a.numpy()[0] for a in self.dtmin.values()))

    def step(self, dt: float, t: float) -> None:
        K, ng, U = self.K, self.ng, self.s.U
        for sid, d in self.data.items():
            wp.copy(d.U0, U[sid])
        for stage, (ca, cb, c) in enumerate(SSPRK3):
            self.rhs(t + c * dt, stage)
            for sid, d in self.data.items():
                wp.launch(K["rk"], dim=d.sub.nc,
                          inputs=[U[sid], d.U0, d.R, self.s.metrics[sid].vol, self.prec.scalar(ca),
                                  self.prec.scalar(cb), self.prec.scalar(dt), ng],
                          device=d.sub.device)

    # ---- forcing ---------------------------------------------------------------------------
    def bulk_momentum(self) -> float:
        """Volume average of rho u . direction over all interior cells."""
        K, ng = self.K, self.ng
        total = 0.0
        dvec = self.prec.vec3(*self.force_dir)
        for sid, d in self.data.items():
            dev = d.sub.device
            tmp = wp.empty(d.sub.nc, dtype=wp.float64, device=dev)
            wp.launch(K["momentum"], dim=d.sub.nc, inputs=[self.s.U[sid], dvec, self.s.metrics[sid].vol, ng],
                      outputs=[tmp], device=dev)
            wp.utils.array_sum(tmp.flatten(), out=self._msum[dev])
            total += float(self._msum[dev].numpy()[0])
        return total / self._vol_total

    def apply_mass_flux_control(self, dt: float, t: float) -> None:
        """Restore the bulk momentum to cfg.forcing.target after a step (uniform increment,
        internal energy unchanged); dm/dt is the equivalent mean body force per unit volume."""
        dm = self.s.cfg.forcing.target - self.bulk_momentum()
        dvec = self.prec.vec3(*self.force_dir)
        for sid, d in self.data.items():
            wp.launch(self.K["add_momentum"], dim=d.sub.nc,
                      inputs=[self.s.U[sid], dvec, self.prec.scalar(dm), self.ng], device=d.sub.device)
        self.forcing_history.append((t, dm / dt))

    # ---- wall-modeled faces: forces and surface statistics ----------------------------------
    def _wall_sample(self, accumulate: bool) -> None:
        """Launch wall_sample on every wall-modeled face (W must hold the current primitives)."""
        sc = self.prec.scalar
        p_inf = sc(self.params.w_inf[4])
        for sid, d in self.data.items():
            m = self.s.metrics[sid]
            S = (m.Si, m.Sj, m.Sk)
            for (ax, side), f in d.wm.items():
                if f.force is None:
                    f.force = wp.zeros(f.tw.shape, dtype=self.prec.vec3, device=d.sub.device)
                if f.acc is None:
                    f.acc = wp.zeros(f.tw.shape, dtype=vec5d, device=d.sub.device)
                wp.launch(self.K["wall_sample"], dim=f.tw.shape,
                          inputs=[d.W, S[ax], ax, side, self.ng, d.sub.nc[ax], f.tw, f.q, p_inf, int(accumulate)],
                          outputs=[f.force, f.acc], device=d.sub.device)

    def wall_forces(self) -> np.ndarray:
        """Total force (3,) exerted by the fluid on all wall-modeled faces: modeled shear (as used
        in the last step) plus pressure relative to the free-stream pressure."""
        if not any(d.wm for d in self.data.values()):
            return np.full(3, np.nan)
        self.update_primitives()
        self._wall_sample(accumulate=False)
        return sum(f.force.numpy().astype(np.float64).reshape(-1, 3).sum(axis=0)
                   for d in self.data.values() for f in d.wm.values())

    # ---- statistics ------------------------------------------------------------------------
    def accumulate_stats(self) -> None:
        self.update_primitives()
        for sid, d in self.data.items():
            if sid not in self.stats:
                self.stats[sid] = wp.zeros(d.sub.nc, dtype=vec16d, device=d.sub.device)
            wp.launch(self.K["stats"], dim=d.sub.nc, inputs=[d.W, d.mut, self.ng, self.params],
                      outputs=[self.stats[sid]], device=d.sub.device)
        self._wall_sample(accumulate=True)
        self.stats_count += 1

    def stats_mean(self, sid: int) -> dict[str, np.ndarray]:
        """Time averages per field for one sub-block (interior cells)."""
        acc = self.stats[sid].numpy() / max(self.stats_count, 1)
        return {k: acc[..., i] for i, k in enumerate(STAT_FIELDS)}

    def wall_sums(self) -> dict[tuple[int, int, int], np.ndarray]:
        """(sid, axis, side) -> raw float64 sums (n_t1, n_t2, len(WALL_FIELDS)) of the wall samples."""
        return {(sid, ax, side): f.acc.numpy().reshape(f.acc.shape + (len(WALL_FIELDS),))
                for sid, d in self.data.items() for (ax, side), f in d.wm.items() if f.acc is not None}

    def set_stats_sums(self, count: int, cell_sums: dict[int, np.ndarray],
                       wall_sums: dict[tuple[int, int, int], np.ndarray]) -> None:
        """Restore accumulated (un-normalized) statistics, e.g. from a restart file."""
        self.stats_count = int(count)
        self.stats = {sid: wp.array(np.ascontiguousarray(a, dtype=np.float64), dtype=vec16d,
                                    device=self.data[sid].sub.device) for sid, a in cell_sums.items()}
        for (sid, ax, side), a in wall_sums.items():
            d = self.data[sid]
            d.wm[(ax, side)].acc = wp.array(np.ascontiguousarray(a, dtype=np.float64), dtype=vec5d,
                                            device=d.sub.device)

    def reset_stats(self) -> None:
        self.stats = {}
        self.stats_count = 0
        for d in self.data.values():
            for f in d.wm.values():
                f.acc = None

    def residual_norms(self) -> np.ndarray:
        """RMS of dU/dt (= R/V) over all interior cells, per conservative variable."""
        tot = np.zeros(5)
        n = 0
        for sid, d in self.data.items():
            v = self.s.interior(self.s.metrics[sid].vol.numpy()).astype(np.float64)
            r = d.R.numpy().astype(np.float64) / v[..., None]
            tot += (r**2).sum(axis=(0, 1, 2))
            n += v.size
        return np.sqrt(tot / max(n, 1))
