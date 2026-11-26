"""Solver orchestrator.

Grid loading, decomposition, connectivity, metrics, state allocation, halo exchange,
time integration loop, output and restart.
"""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path

import numpy as np
import warp as wp

from .config import CaseConfig
from .grid.block import SubBlock
from .grid.connectivity import (
    assign_boundary_conditions,
    build_halo_maps,
    build_padded_nodes,
    detect_interfaces,
    parent_blocks_as_subs,
    slice_padded_nodes,
)
from .grid.metrics import Metrics, compute_metrics
from .grid.partition import assign_devices, split_blocks
from .grid.patches import SplitGrid, split_at_patches, unsplit
from .io.cgns import read_cgns
from .io.extrude import extrude_block
from .io.nmf import read_nmf
from .io.output import SnapshotWriter
from .io.plot3d import read_plot3d
from .io.restart import read_restart, write_restart
from .numerics.integrator import Integrator
from .parallel.devices import enable_peer_access, init_warp, resolve_devices, resolve_partitions
from .parallel.halo import HaloExchanger
from .physics.gas import GasModel
from .precision import get_precision

log = logging.getLogger("wmles_wrap")


def load_grid(cfg: CaseConfig) -> SplitGrid:
    """Read, split at patches (NMF/CGNS), scale, extrude and orient (right-handed) the parent
    grid blocks. Returns the blocks with the patch labels of their faces."""
    if cfg.grid.format == "cgns":
        blocks, patches, _ = read_cgns(cfg.grid_path, base=cfg.grid.base)
        grid = split_at_patches(blocks, patches)
    else:
        blocks = read_plot3d(cfg.grid_path)
        if cfg.nmf_path is not None:
            grid = split_at_patches(blocks, read_nmf(cfg.nmf_path, [b.shape[:3] for b in blocks]))
        else:
            grid = unsplit(blocks)
    for b, x in enumerate(grid.blocks):
        if cfg.grid.extrude is not None:
            if x.shape[2] != 1:
                raise ValueError(f"grid.extrude given but block {b} is already 3D {x.shape}")
            x = extrude_block(x, cfg.grid.extrude.nk, cfg.grid.extrude.length)
        if min(x.shape[:3]) < 2:
            raise ValueError(f"block {b} has fewer than 2 nodes in a direction: {x.shape}; "
                             "2D grids need grid.extrude")
        grid.blocks[b] = x * cfg.grid_scale
        if _handedness(grid.blocks[b]) < 0:
            log.warning("block %d is left-handed; reversing k index", b)
            grid.flip_k(b)
    return grid


def check_imported_interfaces(grid: SplitGrid, interfaces) -> None:
    """Every face on a 1-to-1 patch of the grid file must have been matched geometrically."""
    from .config import FACE_NAMES

    connected = {(i.recv, i.recv_face) for i in interfaces}
    missing = sorted(grid.interface_faces - connected)
    if missing:
        raise ValueError("1-to-1 interfaces in the grid file without a geometric match: "
                         + ", ".join(f"block {b} {FACE_NAMES[f]}" for b, f in missing))


def _handedness(x: np.ndarray) -> float:
    e1 = x[1, 0, 0] - x[0, 0, 0]
    e2 = x[0, 1, 0] - x[0, 0, 0]
    e3 = x[0, 0, 1] - x[0, 0, 0]
    return float(np.dot(np.cross(e1, e2), e3))


class Solver:
    def __init__(self, cfg: CaseConfig):
        init_warp()
        self.cfg = cfg
        self.prec = get_precision(cfg.numerics.precision)
        self.ng = cfg.numerics.ghost_layers
        self.gas = GasModel.from_config(cfg)
        self.step = 0
        self.time = 0.0

        t0 = _time.perf_counter()
        self.grid = load_grid(cfg)
        self.parent_blocks = self.grid.blocks
        parents = parent_blocks_as_subs(self.parent_blocks)
        translations = [p.translation for p in cfg.periodic]
        self.interfaces = detect_interfaces(parents, translations, bc_faces=self.grid.bc_faces)  # parent level
        check_imported_interfaces(self.grid, self.interfaces)
        build_padded_nodes(parents, self.interfaces, self.ng)

        self.devices = resolve_devices(cfg.parallel.devices)
        self.peer = enable_peer_access(self.devices)
        nparts = resolve_partitions(cfg.parallel.partitions, self.devices)
        self.subs: list[SubBlock] = split_blocks(self.parent_blocks, nparts, min_cells=max(4, self.ng))
        self.device_load = assign_devices(self.subs, self.devices)
        slice_padded_nodes(self.subs, parents, self.ng)
        assign_boundary_conditions(self.subs, self.interfaces, cfg.boundaries, self.grid.labels)

        self.metrics: dict[int, Metrics] = {
            s.id: compute_metrics(s.xp, self.prec, s.device) for s in self.subs
        }
        self.halo = HaloExchanger(self.subs, build_halo_maps(self.subs, self.interfaces, self.ng))
        self.U = {s.id: self._alloc_state(s) for s in self.subs}
        self.init_freestream()
        self.integrator = Integrator(self)
        if cfg.init.type == "channel":
            self.init_channel()
        self.setup_time = _time.perf_counter() - t0

    # ---- state ------------------------------------------------------------------------------
    def _alloc_state(self, s: SubBlock) -> wp.array:
        return wp.zeros(s.padded_shape_cells(self.ng), dtype=self.prec.vec5, device=s.device)

    def set_state(self, sid: int, U_padded: np.ndarray) -> None:
        s = next(b for b in self.subs if b.id == sid)
        self.U[sid] = wp.array(U_padded.astype(self.prec.np_dtype), dtype=self.prec.vec5, device=s.device)

    def init_freestream(self) -> None:
        rho, u, v, w, T = self.gas.freestream(self.cfg.flow.alpha, self.cfg.flow.beta)
        q = self.gas.conservative(np.float64(rho), u, v, w, np.float64(T))
        for s in self.subs:
            arr = np.broadcast_to(q, s.padded_shape_cells(self.ng) + (5,))
            self.set_state(s.id, np.ascontiguousarray(arr))

    def set_primitive_field(self, fn) -> None:
        """Initialize from ``fn(xc) -> W`` with xc (..., 3) cell centroids and W (..., 5) =
        [rho, u, v, w, p] (nondimensional)."""
        for s in self.subs:
            xc = self.metrics[s.id].xc.numpy().astype(np.float64)
            W = np.asarray(fn(xc), dtype=np.float64)
            rho, p = W[..., 0], W[..., 4]
            T = p / (rho * self.gas.R)
            self.set_state(s.id, self.gas.conservative(rho, W[..., 1], W[..., 2], W[..., 3], T))

    def init_channel(self) -> None:
        """Turbulent-like channel initial condition: 1/7 power-law mean profile with bulk velocity
        1 along forcing.direction (wall-normal = y), random + streak perturbations vanishing at
        the walls, uniform T = 1 and p = R (rho = 1)."""
        ic = self.cfg.init
        d = self.integrator.force_dir
        R = self.gas.R
        self._all_x = np.concatenate([b[..., 0].ravel() for b in self.parent_blocks])
        self._all_z = np.concatenate([b[..., 2].ravel() for b in self.parent_blocks])

        def noise(xc, comp):
            """Deterministic pseudo-random field in [-1, 1] from the cell position, so the initial
            condition does not depend on the partitioning."""
            k = np.array([12.9898, 78.233, 37.719]) * (1.0 + 0.37 * comp) + 0.1 * ic.seed
            v = np.sin(xc @ k * 1e3) * 43758.5453
            return 2.0 * (v - np.floor(v)) - 1.0

        def fn(xc):
            eta = (xc[..., 1] - ic.center) / ic.half_height
            damp = np.clip(1.0 - eta**2, 0.0, 1.0)
            umean = 8.0 / 7.0 * np.clip(1.0 - np.abs(eta), 0.0, 1.0) ** (1.0 / 7.0)
            x, z = xc[..., 0], xc[..., 2]
            lx, lz = max(np.ptp(self._all_x), 1e-12), max(np.ptp(self._all_z), 1e-12)
            A = ic.amplitude
            W = np.zeros(xc.shape[:-1] + (5,))
            W[..., 0] = 1.0
            vel = umean[..., None] * d
            pert = A * damp[..., None] * np.stack([noise(xc, c) for c in range(3)], axis=-1)
            # large-scale streamwise vortices / streaks to accelerate transition
            pert[..., 0] += 2 * A * damp * np.cos(4 * np.pi * z / lz) * np.sin(np.pi * eta)
            pert[..., 1] += A * damp * np.sin(4 * np.pi * x / lx) * np.cos(4 * np.pi * z / lz)
            W[..., 1:4] = vel + pert
            W[..., 4] = R
            return W

        self.set_primitive_field(fn)
        if self.integrator.mass_flux:
            self.integrator.apply_mass_flux_control(1.0, 0.0)
            self.integrator.forcing_history.clear()

    def write_stats(self) -> Path | None:
        """Write time-averaged fields (vtm) and per-parent-block arrays (npz). Wall-modeled faces
        add ``wall{b:04d}_{face}`` (means of WALL_FIELDS), ``wallxf...`` (face centroids) and
        ``wallS...`` (face area vectors pointing into the fluid)."""
        I = self.integrator
        if I.stats_count == 0:
            return None
        from .config import FACE_NAMES
        from .numerics.kernels import STAT_FIELDS, WALL_FIELDS

        blocks = []
        parent = [np.zeros(tuple(n - 1 for n in x.shape[:3]) + (len(STAT_FIELDS),)) for x in self.parent_blocks]
        for s in self.subs:
            m = I.stats_mean(s.id)
            blocks.append((f"block{s.parent}_part{s.id}", s.x, m))
            lo, hi = s.lo, s.hi
            parent[s.parent][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = np.stack([m[k] for k in STAT_FIELDS], -1)
        writer = SnapshotWriter(self.out_dir, basename="stats")
        path = writer.write(self.step, self.time, blocks)
        walls = {}
        for (b, f), a in self._parent_wall_sums().items():
            key = f"{b:04d}_{FACE_NAMES[f]}"
            xf, S = self._wall_face_geometry(b, f)
            walls.update({f"wall{key}": a / I.stats_count, f"wallxf{key}": xf, f"wallS{key}": S})
        np.savez(self.out_dir / "stats.npz", fields=np.array(STAT_FIELDS), count=I.stats_count, step=self.step,
                 time=self.time, **{f"block{b:04d}": a for b, a in enumerate(parent)},
                 **{f"xc{b:04d}": self._parent_centroids(b) for b in range(len(parent))},
                 forcing=np.array(I.forcing_history), wall_fields=np.array(WALL_FIELDS), **walls)
        return path

    # ---- statistics in parent-block layout (stats.npz, restart) -------------------------------
    def _parent_stats_sums(self) -> list[np.ndarray]:
        """Raw (un-normalized) cell statistics sums per parent block."""
        from .numerics.kernels import STAT_FIELDS

        out = [np.zeros(tuple(n - 1 for n in x.shape[:3]) + (len(STAT_FIELDS),)) for x in self.parent_blocks]
        for s in self.subs:
            if s.id in self.integrator.stats:
                lo, hi = s.lo, s.hi
                a = self.integrator.stats[s.id].numpy()
                out[s.parent][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = a.reshape(a.shape[:3] + (len(STAT_FIELDS),))
        return out

    def _wall_slices(self, s: SubBlock, ax: int) -> tuple[slice, slice]:
        """Slices of a sub-block's wall face (tangential cells) in the parent face array."""
        t = [a for a in range(3) if a != ax]
        return slice(s.lo[t[0]], s.hi[t[0]]), slice(s.lo[t[1]], s.hi[t[1]])

    def _parent_face_arrays(self, values: dict[tuple[int, int, int], np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
        """Assemble per-sub-block face arrays {(sid, axis, side): (n_t1, n_t2[, ...])} into
        parent-face arrays {(parent block, face): ...}."""
        out: dict[tuple[int, int], np.ndarray] = {}
        subs = {s.id: s for s in self.subs}
        for (sid, ax, side), a in values.items():
            s = subs[sid]
            key = (s.parent, 2 * ax + side)
            if key not in out:
                pnc = [n - 1 for n in s.parent_nn]
                out[key] = np.zeros(tuple(pnc[t] for t in range(3) if t != ax) + a.shape[2:])
            out[key][self._wall_slices(s, ax)] = a
        return out

    def _parent_wall_sums(self) -> dict[tuple[int, int], np.ndarray]:
        """(parent block, face) -> raw sums (n_t1, n_t2, len(WALL_FIELDS)) on wall-modeled faces."""
        return self._parent_face_arrays(self.integrator.wall_sums())

    def _wall_face_geometry(self, b: int, f: int) -> tuple[np.ndarray, np.ndarray]:
        """Face centroids and area vectors (pointing into the fluid) of a parent block face."""
        x = self.parent_blocks[b]
        ax, side = f // 2, f % 2

        def layer(idx):
            P = np.take(x, idx, axis=ax)
            return P, 0.25 * (P[:-1, :-1] + P[1:, :-1] + P[:-1, 1:] + P[1:, 1:])

        P, xf = layer(0 if side == 0 else x.shape[ax] - 1)
        _, x2 = layer(1 if side == 0 else x.shape[ax] - 2)
        S = 0.5 * np.cross(P[1:, 1:] - P[:-1, :-1], P[:-1, 1:] - P[1:, :-1])
        if np.sum(S * (x2 - xf)) < 0:
            S = -S
        return xf, S

    # ---- force monitor ------------------------------------------------------------------------
    def force_coefficients(self) -> np.ndarray:
        """[Fx, Fy, Fz, CL, CD] on the wall-modeled faces. CL/CD use flow.ref_area (NaN if unset),
        free-stream dynamic pressure 1/2 (rho_inf = |u_inf| = 1), drag along the free stream and
        lift normal to it in the x-y plane."""
        F = self.integrator.wall_forces()
        cl = cd = np.nan
        if self.cfg.flow.ref_area:
            a, b = np.radians(self.cfg.flow.alpha), np.radians(self.cfg.flow.beta)
            qa = 0.5 * self.cfg.flow.ref_area
            cd = float(F @ [np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)]) / qa
            cl = float(F @ [-np.sin(a), np.cos(a), 0.0]) / qa
        return np.array([*F, cl, cd])

    def _log_forces(self) -> str:
        c = self.force_coefficients()
        path = self.out_dir / "forces.dat"
        new = not path.exists()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            if new:
                fh.write("# step time Fx Fy Fz CL CD  (forces on wall-modeled faces, nondimensional)\n")
            fh.write(f"{self.step} {self.time:.10e} " + " ".join(f"{v:.10e}" for v in c) + "\n")
        return f"  CL {c[3]:.5f} CD {c[4]:.5f}" if self.cfg.flow.ref_area else f"  F {c[0]:.4e} {c[1]:.4e}"

    def _take_stats_sample(self) -> bool:
        r = self.cfg.run
        if r.stats_start_time is not None:
            t0 = r.stats_start_time
            return self.step % r.stats_every == 0 and self.time >= t0 - 1e-12 * max(1.0, abs(t0))
        return bool(r.stats_start) and self.step >= r.stats_start and (self.step - r.stats_start) % r.stats_every == 0

    def _parent_centroids(self, b: int) -> np.ndarray:
        x = self.parent_blocks[b]
        return 0.125 * sum(x[i:x.shape[0] - 1 + i, j:x.shape[1] - 1 + j, k:x.shape[2] - 1 + k]
                           for i in (0, 1) for j in (0, 1) for k in (0, 1))

    def advance(self, nsteps: int | None = None, t_end: float | None = None, log_every: int = 0) -> None:
        """Advance ``nsteps`` steps and/or until ``t_end`` (final step clipped to hit t_end)."""
        cfg = self.cfg
        n = 0
        t_wall = _time.perf_counter()
        while (nsteps is None or n < nsteps) and (t_end is None or self.time < t_end - 1e-14 * max(1.0, t_end)):
            dt = cfg.run.dt if cfg.run.dt is not None else self.integrator.compute_dt(cfg.numerics.cfl)
            if t_end is not None:
                dt = min(dt, t_end - self.time)
            self.integrator.step(dt, self.time)
            self.time += dt
            self.step += 1
            n += 1
            if self.integrator.mass_flux:
                self.integrator.apply_mass_flux_control(dt, self.time)
            if self._take_stats_sample():
                self.integrator.accumulate_stats()
            if log_every and self.step % log_every == 0:
                res = self.integrator.residual_norms()
                el = _time.perf_counter() - t_wall
                ncell = sum(s.ncells for s in self.subs)
                extra = ""
                if self.integrator.forcing_history:
                    extra = f"  forcing {self.integrator.forcing_history[-1][1]:.4e}"
                if any(d.wm for d in self.integrator.data.values()):
                    extra += f"  u_tau(wm) {self.integrator.wall_friction_velocity():.5f}"
                    extra += self._log_forces()
                log.info("step %8d  t %.6e  dt %.3e  res(rho,rhoE) %.3e %.3e  %.3g Mcell-steps/s%s",
                         self.step, self.time, dt, res[0], res[4], ncell * n / el / 1e6, extra)
                if not np.all(np.isfinite(res)):
                    raise FloatingPointError(f"non-finite residual at step {self.step}")
            if nsteps is None and t_end is None:
                break

    def run(self) -> None:
        """Run according to cfg.run: steps/t_end, output, restart cadence."""
        cfg = self.cfg.run
        forces = self.out_dir / "forces.dat"
        if self.step == 0:  # fresh run: start a new force history
            forces.unlink(missing_ok=True)
        elif forces.exists():  # restart: append, dropping rows written after the restart step
            keep = [ln for ln in forces.read_text().splitlines(keepends=True)
                    if ln.startswith("#") or int(ln.split()[0]) <= self.step]
            forces.write_text("".join(keep))
        writer = SnapshotWriter(self.out_dir)
        self.write_output(writer)
        target = cfg.steps if cfg.steps > 0 else None
        chunk_points = [c for c in (cfg.output_every, cfg.restart_every) if c > 0]
        while True:
            done = (target is not None and self.step >= target) or (
                cfg.t_end is not None and self.time >= cfg.t_end - 1e-14 * max(1.0, cfg.t_end))
            if done or (target is None and cfg.t_end is None):
                break
            n = min([c - self.step % c for c in chunk_points] + ([target - self.step] if target else []) or [10**9])
            self.advance(nsteps=n, t_end=cfg.t_end, log_every=cfg.log_every)
            if cfg.output_every and self.step % cfg.output_every == 0:
                self.write_output(writer)
                self.write_stats()
            if cfg.restart_every and self.step % cfg.restart_every == 0:
                self.write_restart(self.out_dir / f"restart_{self.step:08d}.npz")
                if cfg.restart_keep > 0:
                    for old in sorted(self.out_dir.glob("restart_[0-9]*.npz"))[:-cfg.restart_keep]:
                        old.unlink()
        if not cfg.output_every or self.step % cfg.output_every != 0:
            self.write_output(writer)
        self.write_stats()
        self.write_restart(self.out_dir / "restart_final.npz")

    def interior(self, a: np.ndarray) -> np.ndarray:
        g = self.ng
        return a[g:-g, g:-g, g:-g]

    def parent_states(self) -> list[np.ndarray]:
        """Assemble interior conservative states per parent block (float64)."""
        out = [np.zeros(tuple(n - 1 for n in x.shape[:3]) + (5,)) for x in self.parent_blocks]
        for s in self.subs:
            lo, hi = s.lo, s.hi
            out[s.parent][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = self.interior(self.U[s.id].numpy())
        return out

    def load_parent_states(self, states: list[np.ndarray]) -> None:
        for s in self.subs:
            arr = np.zeros(s.padded_shape_cells(self.ng) + (5,))
            lo, hi = s.lo, s.hi
            g = self.ng
            arr[g:-g, g:-g, g:-g] = states[s.parent][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            self.set_state(s.id, arr)
        self.halo.exchange(self.U)

    # ---- I/O --------------------------------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        p = Path(self.cfg.output.dir)
        return p if p.is_absolute() else self.cfg.case_dir / p

    def write_output(self, writer: SnapshotWriter | None = None) -> Path:
        writer = writer or SnapshotWriter(self.out_dir)
        blocks = []
        fields = self.cfg.output.fields
        for s in self.subs:
            prim = self.gas.primitives(self.interior(self.U[s.id].numpy()).astype(np.float64))
            data = {k: prim[k] for k in fields if k in prim}
            data["volume"] = self.interior(self.metrics[s.id].vol.numpy()).astype(np.float64)
            blocks.append((f"block{s.parent}_part{s.id}", s.x, data))
        return writer.write(self.step, self.time, blocks)

    def write_restart(self, path: str | Path) -> None:
        """State per parent block, the wall-model state (tau_w, q: the equilibrium model resumes
        warm-started, as in an uninterrupted run) and, once sampling has started, the accumulated
        statistics sums (so long averaging windows survive interruptions)."""
        from .config import FACE_NAMES

        extra = {}
        I = self.integrator
        wm = {(sid, ax, sd): f for sid, d in I.data.items() for (ax, sd), f in d.wm.items()}
        if wm and all(f.warm for f in wm.values()):
            tw = self._parent_face_arrays({k: f.tw.numpy().astype(np.float64) for k, f in wm.items()})
            q = self._parent_face_arrays({k: f.q.numpy().astype(np.float64) for k, f in wm.items()})
            extra.update({f"wmtw{b:04d}_{FACE_NAMES[f]}": a for (b, f), a in tw.items()})
            extra.update({f"wmq{b:04d}_{FACE_NAMES[f]}": a for (b, f), a in q.items()})
        if I.stats_count > 0:
            extra["stats_count"] = I.stats_count
            extra.update({f"stats{b:04d}": a for b, a in enumerate(self._parent_stats_sums())})
            extra.update({f"wallsum{b:04d}_{FACE_NAMES[f]}": a for (b, f), a in self._parent_wall_sums().items()})
        write_restart(path, self.step, self.time, self.parent_states(), self.prec.name, extra)

    def read_restart(self, path: str | Path) -> None:
        from .config import FACE_NAMES

        self.step, self.time, states, extra = read_restart(path, with_extra=True)
        self.load_parent_states(states)
        for d in self.integrator.data.values():
            s = d.sub
            for (ax, side), f in d.wm.items():
                key = f"{s.parent:04d}_{FACE_NAMES[2 * ax + side]}"
                if f"wmtw{key}" in extra:
                    sl = self._wall_slices(s, ax)
                    f.tw = wp.array(np.ascontiguousarray(extra[f"wmtw{key}"][sl]).astype(self.prec.np_dtype),
                                    dtype=self.prec.vec3, device=s.device)
                    f.q = wp.array(np.ascontiguousarray(extra[f"wmq{key}"][sl]).astype(self.prec.np_dtype),
                                   dtype=self.prec.scalar, device=s.device)
                    f.warm = True
        if "stats_count" not in extra:
            return
        cell, wall = {}, {}
        for s in self.subs:
            lo, hi = s.lo, s.hi
            cell[s.id] = extra[f"stats{s.parent:04d}"][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        for sid, d in self.integrator.data.items():
            s = d.sub
            for ax, side in d.wm:
                key = f"wallsum{s.parent:04d}_{FACE_NAMES[2 * ax + side]}"
                if key in extra:
                    wall[(sid, ax, side)] = extra[key][self._wall_slices(s, ax)]
        self.integrator.set_stats_sums(int(extra["stats_count"]), cell, wall)
        log.info("restart: resumed statistics with %d samples", int(extra["stats_count"]))

    # ---- diagnostics ------------------------------------------------------------------------
    def summary(self) -> str:
        from .config import FACE_NAMES

        lines = [
            f"precision        : {self.prec.name}",
            f"devices          : {self.devices}",
            f"parent blocks    : {[tuple(x.shape[:3]) for x in self.parent_blocks]} (nodes)",
        ]
        split = any(o != (b, (0, 0, 0)) for b, o in enumerate(self.grid.origin))
        for b, x in enumerate(self.parent_blocks):
            fb, lo = self.grid.origin[b]
            lab = {FACE_NAMES[f]: "/".join(v) for f, v in sorted(self.grid.labels[b].items())}
            if split or lab:
                rng = ", ".join(f"{l}:{l + n - 1}" for l, n in zip(lo, x.shape[:3]))
                lines.append(f"  parent {b:3d} = file block {fb} nodes [{rng}] patches {lab}")
        lines += [
            f"total cells      : {sum(s.ncells for s in self.subs):,}",
            f"sub-blocks       : {len(self.subs)}",
        ]
        for s in self.subs:
            bcs = {FACE_NAMES[f]: b.type for f, b in s.face_bc.items()}
            lines.append(f"  sub {s.id:3d} parent {s.parent} cells {s.nc} on {s.device} bc {bcs}")
        lines.append(f"device load      : {self.device_load}")
        nper = sum(1 for i in self.interfaces if i.periodic)
        lines.append(f"block interfaces : {len(self.interfaces)} one-sided ({nper} periodic)")
        gcl = max(m.gcl_max() for m in self.metrics.values())
        vmin = min(m.min_volume(self.ng) for m in self.metrics.values())
        lines.append(f"GCL max residual : {gcl:.3e} (relative)")
        lines.append(f"min cell volume  : {vmin:.6e}")
        lines.append(f"setup time       : {self.setup_time:.2f} s")
        return "\n".join(lines)
