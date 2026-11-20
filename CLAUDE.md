# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Compressible wall-modeled LES solver: explicit, cell-centered, multiblock finite volume, written in Python + NVIDIA Warp, single-process multi-GPU (one process drives N CUDA devices). Cases are driven by a YAML file (see [cases/channel/case.yaml](cases/channel/case.yaml)); output is PyVista `.vtm` + `.pvd`. The roadmap (Phase 0 infrastructure → 1 code verification → 2 SGS/compressible turbulence → 3 wall-model validation → OAT15A/CRM transonic buffet) lives in `~/.claude/plans/i-would-like-to-dreamy-lerdorf.md`. Phase 0 (infrastructure) is done; Phase 1 (numerics + code verification) is on branch `phase1-numerics`. Verification results and scripts: [verification/README.md](verification/README.md). Wall-modeled walls currently behave as no-slip walls (`bc_code` in `numerics/integrator.py`); SGS models are not implemented yet (`mut` array is zero).

## Environment and commands

Always use the conda env `pyvista` (Warp, PyVista, PyYAML, SciPy, pytest, `plot3d` installed; no mpi4py/h5py). The package is not installed; tests use `pythonpath = ["src"]`, CLI runs need `PYTHONPATH=src`.

```bash
conda run -n pyvista python -m pytest -q -p no:warnings                    # all tests
conda run -n pyvista python -m pytest -q tests/test_grid.py::test_unit_cube_metrics   # single test
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap info cases/channel/case.yaml [--write]
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap run cases/channel/case.yaml
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap extrude in.p3d out.p3d --nk 49 --length 4.8
conda run -n pyvista env PYTHONPATH=src python verification/mms_order.py   # also shock_tube.py, couette.py, tgv.py
```

The dev machine is macOS with a **CPU-only Warp build**; production is 8× L40S (weak FP64, PCIe, no NVLink). Develop and verify on CPU in float64; production uses `numerics.precision: float32`. Use `parallel.partitions: N` on CPU to exercise the decomposition and halo code.

## Architecture (things that span files)

- **Index conventions** ([grid/block.py](src/wmles_wrap/grid/block.py)): logical cell `c` lies between nodes `c` and `c+1`. Device arrays are padded with `ng` (≥2) ghost layers, so padded = logical + ng. Faces are numbered 0..5 = imin, imax, jmin, jmax, kmin, kmax (axis = f//2, side = f%2).
- **Setup pipeline** ([solver.py](src/wmles_wrap/solver.py)):
  1. Read Plot3D, optionally extrude, scale by `1/ref_length`, flip k for left-handed blocks.
  2. Detect connectivity at the **parent-block** level.
  3. Build parent padded nodes (mirror at physical boundaries, donor copy minus periodic offset at interfaces).
  4. Split into sub-blocks and assign them to devices.
  5. Sub-block padded nodes are windows of the parent padded nodes.
  6. Compute metrics on each device.
  7. Build halo maps.
- **Connectivity** ([grid/connectivity.py](src/wmles_wrap/grid/connectivity.py)): geometric full-face matching with a CGNS-like transform `idx_d = M @ (idx_r - A0) + B0` (M is a signed permutation), extended past the face to map ghosts. Cells map through half-integer node positions. Periodicity is purely geometric: faces that match under any `periodic[].translation` are connected. Halo maps send ghost cells through the parent transform, and each donor cell is owned by the sub-block whose interior contains it, so the partitioner never has to match partial faces. Edge/corner ghosts whose donor cell is itself a ghost go into exchange stage g = the number of directions in which the donor cell is a ghost. Stages 1 and 2 run after the BCs, so results are bitwise independent of the partitioning (tested).
- **Numerics** ([numerics/kernels.py](src/wmles_wrap/numerics/kernels.py), [numerics/integrator.py](src/wmles_wrap/numerics/integrator.py)): all kernels come from `make_kernels(prec)`, with runtime constants in a `Params` struct. Each RK stage runs: U→W (primitives, interior) → halo stage 0 → BC kernels (per face, fixed axis order, full padded tangential range) → halo stages 1, 2 → Green-Gauss gradients + sensor (padded range minus 1) → face fluxes per axis (KEEP-PE + sensor-weighted Roe/Rusanov + viscous with directional face-gradient correction) → residual (deterministic face-difference, no atomics) → SSP-RK3 update. Dissipation is off on wall/slip faces (`nodiss`). Dirichlet BCs and MMS sources are evaluated on the host (verification only).
- **Halo exchange** ([parallel/halo.py](src/wmles_wrap/parallel/halo.py)): one index list per (donor, recv) pair. On the same device it uses a direct gather/scatter kernel; across devices it packs, does `wp.copy` (P2P if enabled), then unpacks. `force_staged=True` tests the cross-device path on one device. It is currently synchronous; overlap with streams is planned.
- **Precision**: kernels come from factories cached on a `Precision` ([precision.py](src/wmles_wrap/precision.py)). Modules that define Warp kernels with closure-captured types must **not** use `from __future__ import annotations`: Warp can't resolve string annotations against closures.
- **Nondimensionalization** ([config.py](src/wmles_wrap/config.py)): reference ρ, U = M·a_ref, T_ref, L_ref, so `p = ρT/(γM²)` and μ* = μ(T)/μ(T_ref)/Re. The state is conservative `vec5` [ρ, ρu, ρv, ρw, ρE].
- **Config**: dataclasses with strict validation (unknown keys are errors). BCs attach per parent face name (optionally per `block`); every non-interface parent face must have one.
- **Restart** files store per-*parent*-block interior states, so a run can restart with a different partitioning or device count.

## Grids

[cases/channel/](cases/channel/) holds the UMD WMLES workshop Re_τ=5200 channel grids (2D, single block, Fortran-unformatted LE float64). Extrude in z to 4.8h with nk=49 (handled via `grid.extrude` in the YAML). `extrude.f90` is the original reference tool.
