# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Compressible wall-modeled LES solver: explicit, cell-centered, multiblock finite volume, written in Python + NVIDIA Warp, single-process multi-GPU (one process drives N CUDA devices). Cases are driven by a YAML file (see [cases/channel/case.yaml](cases/channel/case.yaml)); output is PyVista `.vtm` + `.pvd`. The roadmap (Phase 0 infrastructure → 1 code verification → 2 SGS/compressible turbulence → 3 wall-model validation → OAT15A/CRM transonic buffet) and the status of each phase live in [docs/plan.md](docs/plan.md); keep it updated when a phase item is completed. Phases 0–2 are done (infrastructure, numerics + code verification, SGS/turbulence). Phase 3 (wall models and their validation) is in progress (merged into `main`; remote `origin` = github.com/EduardoMolina/wmles-warp). Verification results and scripts are in [verification/README.md](verification/README.md). The RANS-SA extension (steady RANS, URANS, DDES, mesh deformation, adjoint as a target, chimera grids; stages R0–R10) is planned in [docs/plan_rans.md](docs/plan_rans.md); its design rules for adjoint readiness apply to any new residual code.

## Environment and commands

Always use the conda env `pyvista` (note: `conda run` neither forwards stdin nor streams stdout, so for long background runs call `/opt/miniconda3/envs/pyvista/bin/python` directly) (Warp, PyVista, PyYAML, SciPy, pytest, `plot3d`, `h5py` for CGNS/HDF5 installed; no mpi4py). The package is not installed; tests use `pythonpath = ["src"]`, CLI runs need `PYTHONPATH=src`.

```bash
conda run -n pyvista python -m pytest -q -p no:warnings                    # all tests
conda run -n pyvista python -m pytest -q tests/test_grid.py::test_unit_cube_metrics   # single test
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap info cases/channel/case.yaml [--write]
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap run cases/channel/case.yaml
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap extrude in.p3d out.p3d --nk 49 --length 4.8
conda run -n pyvista env PYTHONPATH=src python verification/mms_order.py   # also shock_tube.py, couette.py, tgv.py
conda run -n pyvista env PYTHONPATH=src python verification/tgv.py --re 1600 --n 64 --t-end 20 --sgs vreman --dissipation roe
conda run -n pyvista env PYTHONPATH=src python verification/channel_profiles.py out/stats.npz --reynolds 2800 --mach 0.2 --plot
python verification/benchmark.py --n 256 --devices 1,2,4,8 --precision float32   # on the GPU node
bash verification/channel_sweep.sh   # GPU node: Re_tau=5200 WMLES sweep, one case per GPU, in runs/ (DRY=1 to only generate)
python verification/tgv.py --n 128 --re 1600 --t-end 20 --sgs vreman --dissipation roe --devices cuda:0 --precision float32
conda run -n pyvista env PYTHONPATH=src python verification/channel_profiles.py out/stats.npz --reynolds 125000 --mach 0.2 \
    --utau forcing --lm cases/channel/LM_Channel_5200_mean_prof.dat --plot   # WMLES channel vs DNS
PYTHONPATH=src python -m wmles_wrap run cases/a_airfoil/case.yaml   # GPU node: A-airfoil WMLES (36.7M cells)
conda run -n pyvista env PYTHONPATH=src python verification/airfoil_surface.py cases/a_airfoil/out/stats.npz \
    --case cases/a_airfoil/case.yaml --forces cases/a_airfoil/out/forces.dat --ref "surface data.dat" --plot
```

The dev machine is macOS with a **CPU-only Warp build**; production is 8× L40S (weak FP64, PCIe, no NVLink). Develop and verify on CPU in float64; production uses `numerics.precision: float32`. Use `parallel.partitions: N` on CPU to exercise the decomposition and halo code.

## Architecture (things that span files)

- **Index conventions** ([grid/block.py](src/wmles_wrap/grid/block.py)): logical cell `c` lies between nodes `c` and `c+1`. Device arrays are padded with `ng` (≥2) ghost layers, so padded = logical + ng. Faces are numbered 0..5 = imin, imax, jmin, jmax, kmin, kmax (axis = f//2, side = f%2).
- **Setup pipeline** ([solver.py](src/wmles_wrap/solver.py)):
  1. Read the grid (Plot3D binary/formatted, optionally with a NASA `.nmf`, or structured CGNS in HDF5 or ADF), split blocks at patch boundaries ([grid/patches.py](src/wmles_wrap/grid/patches.py)), optionally extrude, scale by `1/ref_length`, flip k for left-handed blocks (swapping kmin/kmax patch labels).
  2. Detect connectivity at the **parent-block** level.
  3. Build parent padded nodes (mirror at physical boundaries, donor copy minus periodic offset at interfaces).
  4. Split into sub-blocks and assign them to devices.
  5. Sub-block padded nodes are windows of the parent padded nodes.
  6. Compute metrics on each device (differenced in float64 from float64 coordinates, stored in the run precision).
  7. Build halo maps.
- **Connectivity** ([grid/connectivity.py](src/wmles_wrap/grid/connectivity.py)): geometric full-face matching with a CGNS-like transform `idx_d = M @ (idx_r - A0) + B0` (M is a signed permutation), extended past the face to map ghosts. Cells map through half-integer node positions. Periodicity is purely geometric: faces that match under any `periodic[].translation` are connected. Halo maps send ghost cells through the parent transform, and each donor cell is owned by the sub-block whose interior contains it, so the partitioner never has to match partial faces. Edge/corner ghosts whose donor cell is itself a ghost go into exchange stage g = the number of directions in which the donor cell is a ghost. Stages 1 and 2 run after the BCs, so results are bitwise independent of the partitioning (tested).
- **Numerics** ([numerics/kernels.py](src/wmles_wrap/numerics/kernels.py), [numerics/integrator.py](src/wmles_wrap/numerics/integrator.py)): all kernels come from `make_kernels(prec)`, with runtime constants in a `Params` struct. Each RK stage runs: U→W (primitives, interior) → halo stage 0 → BC kernels (per face, fixed axis order, full padded tangential range) → halo stages 1, 2 → Green-Gauss gradients + sensor (padded range minus 1) → face fluxes per axis (KEEP-PE + sensor-weighted Roe/Rusanov + viscous with directional face-gradient correction) → residual (deterministic face-difference, no atomics) → SSP-RK3 update. Dissipation is off on wall/slip faces (`nodiss`). Dirichlet BCs and MMS sources are evaluated on the host (verification only).
- **SGS / forcing / statistics**: `k_sgs` (Vreman, WALE, Smagorinsky; Δ = V^(1/3)) fills `mut` after the gradients; μ_t is zeroed on wall/slip faces. Constant mass flux is applied *after* each step as a uniform momentum increment that keeps internal energy (`apply_mass_flux_control`); its equivalent force goes into `forcing_history`. Statistics are float64 sums of `STAT_FIELDS` per cell (`k_stats`), written by `Solver.write_stats` to `stats.vtm` + `stats.npz` (per parent block, with centroids and forcing history). Global sums (mass flux) depend on summation order, so forced runs are only partition-independent to round-off. Wall-modeled faces also accumulate float64 sums of first-cell p, τ_w (vec3) and q (`k_wall_sample`), written as `wall{b}_{face}` (+ `wallxf`, `wallS` = face centroids and area vectors into the fluid) in `stats.npz`; the same kernel gives the force monitor (`forces.dat`, C_L/C_D with `flow.ref_area`) every `log_every` steps. `run.stats_start_time` starts sampling at a time instead of a step.
- **Wall models** (`k_wall_model` in [numerics/kernels.py](src/wmles_wrap/numerics/kernels.py), numpy reference in [wallmodel/reference.py](src/wmles_wrap/wallmodel/reference.py), and tests require the two to agree to 1e-10, so change both together). `wall_modeled` faces use slip-mirror ghosts (impermeable inviscid flux). Their viscous flux is replaced in `k_flux` by the modeled traction τ_w·t̂·|S| and heat flux Q·|S|, with Q = k ∂T/∂n and n into the fluid; the sign flips on the high side. There is one thread per wall face. The equilibrium ODE model integrates momentum and energy in closed form (quadratures on a stretched 1D grid, ≤ 64 points in local `vecN` arrays) with an undamped fixed point, warm-started from the stored τ_w. Wall models are evaluated once per time step by default (`wall_model.update`). The exchange point is `cell_index` (1-based from the wall) or `distance`. With `wall_model.type: none`, `wall_modeled` faces fall back to no-slip walls.
- **Warp loop variables**: a variable mutated inside a dynamic loop must not be initialized from a literal or closure constant. Write `m = int(0)` / `y = scalar(0.0)`, not `m = 0` / `y = zero`.
- **Warp literal pitfall**: inside kernels, Python float literals are float32. `scalar(2.0 / 3.0)` written in a kernel loses precision in float64 kernels, so define fractional constants in the factory closure (see the top of `make_kernels`).
- **Halo exchange** ([parallel/halo.py](src/wmles_wrap/parallel/halo.py)): one index list per (donor, recv) pair. On the same device it uses a direct gather/scatter kernel; across devices it packs, does `wp.copy` (P2P if enabled), then unpacks. `force_staged=True` tests the cross-device path on one device. It is currently synchronous; overlap with streams is planned.
- **Precision**: kernels come from factories cached on a `Precision` ([precision.py](src/wmles_wrap/precision.py)). Modules that define Warp kernels with closure-captured types must **not** use `from __future__ import annotations`: Warp can't resolve string annotations against closures.
- **Nondimensionalization** ([config.py](src/wmles_wrap/config.py)): reference ρ, U = M·a_ref, T_ref, L_ref, so `p = ρT/(γM²)` and μ* = μ(T)/μ(T_ref)/Re. The state is conservative `vec5` [ρ, ρu, ρv, ρw, ρE].
- **Config**: dataclasses with strict validation (unknown keys are errors). BCs attach per parent face name and/or grid-file patch label (`boundaries[].patches`), optionally per `block` (indices after patch splitting); the last matching entry wins, and every non-interface parent face must have one.
- **Patches** ([grid/patches.py](src/wmles_wrap/grid/patches.py)): CGNS/NMF sub-face BCs and 1-to-1 interfaces are made full-face by cutting blocks at every patch boundary and carrying the cuts through the 1-to-1 transforms to a fixed point. Interfaces are still found geometrically; imported 1-to-1 faces without a geometric match are an error. Faces with a BC patch are only connected through periodicity (coincident BC faces are a zero-thickness wall, as in TMR 2DCJ).
- **Restart** files store per-*parent*-block interior states, plus the wall-model state (τ_w, q, so the equilibrium model resumes warm-started) and, once sampling has started, the statistics sums, so a run can restart with a different partitioning or device count without losing its averages (`run.restart_keep` prunes old periodic restarts).

## Grids

[cases/channel/](cases/channel/) holds the UMD WMLES workshop Re_τ=5200 channel grids (2D, single block, Fortran-unformatted LE float64). Extrude in z to 4.8h with nk=49 (handled via `grid.extrude` in the YAML). `extrude.f90` is the original reference tool.

[cases/a_airfoil/](cases/a_airfoil/) is the A-airfoil near stall (α = 13.3°, Re_c = 1e7, M = 0.15; reference WRLES Tamaki & Kawai 2023). The grid `a_airfoil_C.xyz` (900 MB, raw multiblock Plot3D, git-ignored) is a 4-block C-mesh: block 0 lower wake, 1 lower surface, 2 upper surface, 3 upper wake (blocks 0/3 share the wake cut on jmin), spanwise periodic in k (z, 0.048c). It only fits the GPU node (~15 GB device memory in float32); test on CPU with a coarsened copy that keeps the interfaces point-matched: `python cases/a_airfoil/coarsen.py [--si 4 --sj 2 --sk 9 --partitions 6]` writes `a_airfoil_coarse.p3d` and `case_coarse.yaml` (both git-ignored).
