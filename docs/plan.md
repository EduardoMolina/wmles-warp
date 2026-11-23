# Plan: compressible WMLES solver in NVIDIA Warp

This document holds the roadmap and the status of each phase. Detailed verification numbers are in
[verification/README.md](../verification/README.md). Code conventions are in [CLAUDE.md](../CLAUDE.md).

## Goal

The solver is a modular, explicit, cell-centered, multiblock finite-volume compressible WMLES solver,
written in Python and NVIDIA Warp. One process drives 1 to 8 GPUs on a single node. Runs are driven by
a YAML input file and write PyVista output. The solver is verified in phases, and the end goal is
transonic buffet on the NASA Common Research Model (CRM).

## Status

| Phase | Content | Status | Branch / commit |
|---|---|---|---|
| 0 | Infrastructure | done | `main` `cb1c8ad` |
| 1 | Code verification | done; a few items deferred (see Phase 1) | `c215c1c` |
| 2 | SGS and compressible turbulence | done on CPU; converged runs need the GPU node | `c1e232e` |
| 3 | Wall-model validation | in progress: models and channel at Re_τ = 5200 done; A-airfoil set up | `main` `3077414` |
| 4 | CGNS, OAT15A buffet, CRM | not started | — |
| R | RANS-SA (steady RANS, URANS, DDES, adjoint target) | planned, see [plan_rans.md](plan_rans.md) | — |

## Environment and design drivers

**Development machine**
- macOS ARM with the conda env `pyvista`: Python 3.14, Warp 1.19 (CPU-only build), PyVista,
  PyYAML, SciPy, pytest, `plot3d` and `h5py` (CGNS/HDF5 input).
- No mpi4py or CUDA.
- Verification runs here in float64. `parallel.partitions: N` exercises the same decomposition and
  halo code that the multi-GPU runs use.

**Production machine**
- AWS g6e.48xlarge with 8× L40S (48 GB each, PCIe Gen4, no NVLink).
- AWS g5 instances with A10G (24 GB, 600 GB/s) for single-GPU and smaller runs.
- FP64 runs at about 1/64 of FP32 speed, so production uses `numerics.precision: float32`.
- Statistics accumulators stay in float64.

**User decisions**
- Single-process multi-device.
- KEEP-PE split-form central flux with sensor-gated upwind dissipation.
- Two wall models, equilibrium ODE (the default) and algebraic, selectable in YAML.

## Architecture (as implemented)

```
src/wmles_wrap/
  cli.py, __main__.py    # python -m wmles_wrap info|run|extrude
  config.py              # YAML -> strictly validated dataclasses, nondimensionalization
  precision.py           # Precision (float32|float64) and cached kernel factories
  solver.py              # setup pipeline, time loop, output/stats/restart cadence
  io/plot3d.py           # Plot3D reader/writer (Fortran/raw, endianness, fp32/64, 2D/3D, iblank)
  io/extrude.py          # port of cases/channel/extrude.f90
  io/output.py           # PyVista .vtm + .pvd time series
  io/restart.py          # per-parent-block .npz (restart works with any partitioning)
  grid/block.py          # SubBlock, index conventions, ghost layers
  grid/connectivity.py   # parent-level face matching, periodicity, halo maps (stages 0/1/2)
  grid/partition.py      # recursive bisection + LPT device assignment
  grid/metrics.py        # face areas, volumes, centroids, GCL check
  parallel/devices.py    # device resolution, peer access
  parallel/halo.py       # gather/scatter on one device; pack -> wp.copy -> unpack across devices
  physics/gas.py         # ideal gas, Sutherland/power/constant viscosity
  numerics/kernels.py    # all Warp kernels: primitives, BCs, gradients + sensor, SGS, fluxes,
                         #   residual, SSP-RK3, dt, forcing, statistics, wall models
  numerics/integrator.py # RK stage sequence, BC codes, wall-model faces, mass-flux control, stats
  wallmodel/reference.py # numpy reference of the wall models (the kernels must match it to 1e-10)
  verify/mms.py, riemann.py  # manufactured solutions, exact Riemann solver
verification/            # study scripts; results in verification/results/
cases/                   # channel (Re_τ = 5200 UMD grids), channel_retau180, a_airfoil
```

**Differences from the original design**
- All kernels live in a single factory, `numerics/kernels.py`. There are no separate inviscid,
  viscous or SGS modules. The BC, SGS and statistics kernels are in the factory too, and the `bc/`
  and `stats/` packages are empty.
- The equilibrium wall model integrates its ODEs once, in closed form, by quadrature: there is no
  TDMA solve. It uses an undamped fixed point warm-started from the stored τ_w and is evaluated once
  per step by default.
- The default shock sensor is `ducros_pressure`: a regularized Ducros sensor multiplied by a Jameson
  pressure/density second-difference sensor. The plain Ducros sensor makes the scheme first order
  in smooth irrotational flow.
- Connectivity is detected on the parent blocks, before splitting. Sub-block halo maps are derived
  from it, so the partitioner never has to match partial faces.
- The halo exchange is still synchronous. Overlapping it with compute (streams/events) is still to
  do.
- Surface output is time-averaged only: wall pressure, modeled τ_w and q on wall-modeled faces in
  `stats.npz`, plus a C_L/C_D history in `forces.dat`. Probes and instantaneous surface output are
  still to do; they are needed for buffet spectra.

## Phase 0: infrastructure — done

- Config loader, Plot3D I/O, extrusion, metrics, PyVista output, restart and CLI.
- Partitioner and halo exchange.
- Exit criteria met:
  - the grid read matches the `plot3d` package;
  - GCL residual is about 1e-17;
  - `.vtm` output opens in PyVista.

## Phase 1: code verification — done

All results are in [verification/README.md](../verification/README.md#phase-1-code-verification).

| Check | Result |
|---|---|
| Free-stream preservation (curvilinear, reoriented multiblock, partitions) | ≤ 1.2e-14 |
| MMS (NS, Euler, NS with Roe + sensor) | L2 order ≈ 2 for all variables |
| Shock tubes (Sod, Lax, 1000:1) | stable |
| Couette with heating | u exact; T converges at second order |
| Inviscid TGV | KEEP-PE kinetic-energy behaviour correct; conservation to 1e-14 |
| Partition independence (1/3/5 partitions, staged halos) | bitwise |

Deferred:
- isentropic vortex long-time transport;
- laminar compressible flat plate;
- Shu–Osher / 2D Riemann problems;
- CPU vs GPU comparison (needs the g6e node).

## Phase 2: SGS and compressible turbulence — done on CPU

- SGS models: Vreman (default, c = 0.07), WALE and Smagorinsky, with Δ = V^(1/3). The kernels match
  a numpy reference to 1e-12.
- TGV at Re = 1600, 32³ and 64³: the results move toward the DNS with refinement. A converged run
  needs 128³ on the GPU.
- Decaying isotropic turbulence at Mt = 0.6: stable with shocklets, and the sensor deactivates with
  grid refinement.
- Wall-resolved channel at Re_τ = 180: the pipeline works, and the momentum balance closes
  (u_τ from the wall shear equals u_τ from the forcing).

To do on the GPU node:
- TGV at 128³;
- converged Re_τ = 180 channel statistics (~3×10⁵ steps);
- supersonic isothermal channel, M = 1.5, Re_τ ≈ 220 (Coleman et al. 1995);
- `verification/benchmark.py` (throughput, strong/weak scaling, P2P vs staged halos, fp32 vs fp64).

## Phase 3: wall-model validation — in progress

**Done**

| Item | Result |
|---|---|
| Equilibrium ODE wall model (compressible momentum + energy, mixing length with damping) | implemented |
| Algebraic wall model (Reichardt + Crocco–Busemann) | implemented |
| Exchange point by cell index or distance | implemented |
| Warp kernel vs numpy reference | agree to 1e-10 |
| A-priori test against Lee & Moser Re_τ = 5200 | equilibrium +0.01…+1.6% (h = 0.05…0.3); algebraic −3.0…−1.6% |
| Crocco–Busemann limit (Pr = Pr_t = 1, M up to 3) | exact |
| Wall traction vs mass-flux forcing | equal to 1e-10 |

**WMLES channel at Re_τ = 5200**

Setup: UMD ar2 grid, 64×32×48 cells; exchange point at cell 3; M = 0.2; 60,000 steps.

| Model | u_τ error vs DNS |
|---|---|
| Equilibrium ODE | −1.9% |
| Algebraic | −4.3% |

Both runs show the log-layer mismatch typical of coarse grids. The outer region matches the DNS.

**GPU sweep** (1× A10G, float32, 200,000 steps, statistics from step 50,000; about 70 Mcell-steps/s
on the ar1 grid):

| Case | u_τ error vs DNS |
|---|---|
| ar2, equilibrium, cell 3 | −2.2% (float64 CPU: −1.9%, so float32/GPU agrees) |
| ar1, equilibrium, cells 1 / 2 / 3 / 4 / 5 | −15.9% / −8.0% / −3.9% / −1.4% / −0.04% |
| ar1, algebraic, cell 3 | −6.1% |

- The exchange location dominates the error (log-layer mismatch). Cells 4–5 (h ≈ 0.10–0.13,
  h⁺ ≈ 510–700, i.e. h ≈ 0.1δ) are within 1.5% of the DNS.
- The finer streamwise grid (ar1) costs about 2 points for both models at cell 3, a resolution
  effect rather than a model effect.
- Plot: `verification/results/channel_re5200_exchange_sensitivity.png`.

**Remaining**
1. **Channel at Re_τ = 5200:**
   - the other UMD resolutions;
   - mean and stress profiles of the ar1 sweep (bring back `stats.npz`);
   - decide the default exchange location (currently cell 3).
2. **Compressible wall-model check:** high-Re supersonic channel (Modesti & Pirozzoli 2016) or a
   heated isothermal-wall case. This validates the energy equation and q_w in the solver.
3. **Zero-pressure-gradient turbulent boundary layer** with recycling/rescaling or synthetic inflow:
   Cf(Re_θ) against correlations and DNS.
4. **Near-stall airfoil: A-airfoil** (step between the channel and the OAT15A buffet case). See
   the section below.
5. **Separation and shock:**
   - Bachalo–Johnson transonic axisymmetric bump (shock-induced separation, the direct precursor
     to buffet);
   - optionally, the NASA wall-mounted hump.

### A-airfoil near stall — set up, production run pending

Why this case: it is the first external-aerodynamics case. It tests the far-field BCs on a
multiblock C-mesh with a wake cut, and the wall model under a strong adverse pressure gradient
with trailing-edge separation, at low Mach and without shocks. That isolates the wall-model
questions from the shock/buffet dynamics of the OAT15A.

- **Conditions:** Aérospatiale A-airfoil, α = 13.3°, Re_c = 1.0×10⁷, M = 0.15, T_∞ = 285 K.
- **Reference:** WRLES by Tamaki & Kawai, AIAA J. 61(2), 698–711 (2023),
  doi:10.2514/1.J062066. The
  [database](https://www.klab.mech.tohoku.ac.jp/database/2022TamakiAIAAJ/README.txt) has Cp, Cf,
  H and Re_θ along the surface, plus velocity/stress profiles and momentum and TKE budgets at
  several x/c.
- **Case:** [cases/a_airfoil/case.yaml](../cases/a_airfoil/case.yaml).
  - Grid `a_airfoil_C.xyz` (not in git): 4-block C-mesh, 36.7 M cells, far field at about 30c,
    spanwise-periodic span 0.048c (81 cells).
  - First wall cell 2.8×10⁻⁴c (y⁺ ≈ 100); wall-tangential spacing 6–8.4×10⁻⁴c.
  - Equilibrium ODE wall model on both surfaces, exchange at cell 3. Vreman SGS, float32.
  - Riemann far field on all outer faces; uniform free-stream start.
- **Run:** 30 CTU (c/U_∞), statistics over t = 20–30 (`run.stats_start_time`).
- **Cost:** dt ≈ 1.4×10⁻⁵ CTU (acoustic CFL at M = 0.15), about 2.1 M steps in total.
  - On one A10G (g5.4xlarge) at about 70 Mcell-steps/s, that is about 13 days, i.e. ~10 h per CTU.
  - GPU memory is about 15 GB (24 GB available).
  - Restarts carry the statistics sums and the wall-model state, so the run can be split into
    jobs and moved to the 8× L40S node.
- **Smoke test (CPU, float64):** grid coarsened 4×2×9 (516 k cells) with
  [cases/a_airfoil/coarsen.py](../cases/a_airfoil/coarsen.py), 6 partitions.
  - Connectivity correct: wake cut and three block joins, plus spanwise periodicity.
  - Stable for 3000 steps (0.1 CTU) of the impulsive start; residuals fall about 20× as the
    start-up pressure pulse leaves. Wall statistics, forces, restart pruning and the
    post-processing script work.
- **Comparison:** `verification/airfoil_surface.py` plots mean Cp and Cf against the WRLES
  surface data, and gives C_L/C_D from the surface integral and from the `forces.dat` history.
- **Watch for:**
  1. The wall model is applied over the whole surface, so the laminar/transitional leading-edge
     region is treated as turbulent. Expect a Cf mismatch near the LE.
  2. Size and onset of the trailing-edge separation on the upper surface.
  3. The span (0.048c) is narrow relative to the trailing-edge boundary layer near stall.
  4. There is no far-field circulation correction at 30c.
- **Possible follow-ups:** exchange location by distance; a sponge layer or vortex-corrected far
  field; a wider span.

## Phase 4 (roadmap): buffet and CRM

- **CGNS reader** (`h5py` is installed in the `pyvista` env; shared with RANS stage R0 in
  [plan_rans.md](plan_rans.md)):
  - Targets structured CGNS/HDF5 exports from ICEM and ANSA.
  - Reads 1-to-1 abutting connectivity from `ZoneGridConnectivity` and boundaries from `ZoneBC`.
  - Splits blocks at sub-face boundaries, so all interfaces are full-face.
  - Supports symmetry-plane and far-field boundaries.
- **Async halo exchange** (streams/events), plus probes and surface/wall-pressure output for
  spectra.
- **Intermediate buffet case: OAT15A airfoil** (M = 0.73, Re_c = 3×10⁶, α ≈ 3.5°) as a
  spanwise-periodic slab. It reuses the airfoil setup validated on the A-airfoil (Phase 3). Validate the buffet frequency (St ≈ 0.07) and the pressure RMS.
- **CRM** (DPW/HLPW structured grids) at transonic buffet conditions: M ≈ 0.85, Re_c ≈ 5×10⁶,
  α beyond buffet onset (about 3.5–5°). Compare with NASA Ames 11-ft and ETW unsteady PSP and
  Kulite data.
- **Memory budget:** about 250 B/cell in fp32, so 8×48 GB holds roughly 1.2×10⁹ cells. That is
  enough for WMLES-class CRM grids (10⁸–10⁹ cells).

## How to run

The commands are in [CLAUDE.md](../CLAUDE.md#environment-and-commands). In short:

```bash
conda run -n pyvista python -m pytest -q -p no:warnings
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap run cases/channel/case.yaml
conda run -n pyvista env PYTHONPATH=src python verification/<study>.py
```

A phase is done when its checks pass and the comparison plots are committed under
`verification/results/`.
