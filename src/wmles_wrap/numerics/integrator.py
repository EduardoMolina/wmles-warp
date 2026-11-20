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
    VISC_CONSTANT,
    VISC_POWER,
    VISC_SUTHERLAND,
    make_kernels,
)

# SSP-RK3 (Shu & Osher): U^{(s)} = a_s U^n + b_s (U^{(s-1)} + dt L(U^{(s-1)})), stage time c_s
SSPRK3 = ((0.0, 1.0, 0.0), (0.75, 0.25, 1.0), (1.0 / 3.0, 2.0 / 3.0, 0.5))

NO_DISSIPATION_BCS = ("wall", "wall_modeled", "slip")


def bc_code(bc) -> int:
    if bc.type in ("wall", "wall_modeled"):
        # TODO(phase 3): wall_modeled applies the wall-model shear stress / heat flux
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
        self.data: dict[int, _SubData] = {sb.id: self._alloc(sb) for sb in solver.subs}
        self.dtmin = {d: wp.zeros(1, dtype=self.prec.scalar, device=d) for d in solver.devices}

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
            code = bc_code(bc)
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

    def rhs(self, t: float) -> None:
        K, ng = self.K, self.ng
        self.update_primitives()
        self.fill_ghosts(t)
        force = self.prec.vec3(*self.force)
        for sid, d in self.data.items():
            m = self.s.metrics[sid]
            dev = d.sub.device
            pad = d.sub.padded_shape_cells(ng)
            wp.launch(K["gradients"], dim=tuple(n - 2 for n in pad),
                      inputs=[d.W, m.Si, m.Sj, m.Sk, m.vol, self.params], outputs=[d.GU, d.GT, d.th], device=dev)
            S = (m.Si, m.Sj, m.Sk)
            for ax in range(3):
                Fa = d.F[ax]
                wp.launch(
                    K["flux"],
                    dim=Fa.shape,
                    inputs=[d.W, d.GU, d.GT, d.th, d.mut, S[ax], m.xc, ax, ng, Fa.shape[ax],
                            d.nodiss[ax][0], d.nodiss[ax][1], self.params],
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
        for ca, cb, c in SSPRK3:
            self.rhs(t + c * dt)
            for sid, d in self.data.items():
                wp.launch(K["rk"], dim=d.sub.nc,
                          inputs=[U[sid], d.U0, d.R, self.s.metrics[sid].vol, self.prec.scalar(ca),
                                  self.prec.scalar(cb), self.prec.scalar(dt), ng],
                          device=d.sub.device)

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
