# Plan: RANS-SA extension (steady RANS, URANS, DDES)

This document holds the roadmap for adding Spalart–Allmaras (SA) RANS modes to the WMLES solver. The
LES roadmap is in [plan.md](plan.md), and code conventions are in [CLAUDE.md](../CLAUDE.md).

## Goal

- Add steady RANS, URANS and DDES modes with the SA model, in the same package as the WMLES solver.
- The modes reuse the grid, connectivity, halo, BC, I/O, `Params` and `make_kernels` flux functions.
- The mode is selected by `solver.mode: les | rans | urans | ddes`.
- The production target is the transonic NASA Common Research Model (CRM).
- Mesh deformation comes before the adjoint. It starts with jig → flight shape and then becomes a
  transposable surface → volume operator.
- Gradient-based shape optimization (discrete adjoint) is a future target. The design decisions
  that enable it are taken from the first stage on.
- Chimera (overset) grids are the last item, for wing/fuselage/tail configurations.

## Status

| Stage | Content | Status |
|---|---|---|
| R0 | Prerequisites: FP64 metrics, line-aware partitioner, GPU wall distance, CGNS reader | not started |
| R1 | Pseudo-time implicit core (mixed-precision defect correction, line relaxation) | not started |
| R2 | SA model (negative-SA) | not started |
| R3 | TMR verification: flat plate (2DZP), coflowing jet (2DCJ), bump (2DB), then NACA 0012 | not started; 2DZP/2DCJ/2DB grids and SA data in `cases/tmr/` |
| R4 | Acceleration: FGMRES and/or FAS multigrid | not started |
| R5 | Production steady RANS: NASA CRM (DPW-5/6) | not started |
| R6 | Dual-time stepping (URANS) | not started |
| R7 | DDES / IDDES | not started |
| R8 | Mesh deformation: jig → flight mesh (twist and bending about an elastic axis), then general surface → volume | not started |
| R9 | Discrete adjoint and shape gradients | target; design rules apply from R0 |
| R10 | Chimera (overset) grids for wing/fuselage/tail configurations | not started |

## Hardware

| Instance | GPU | Memory | Bandwidth | FP32 | FP64 |
|---|---|---|---|---|---|
| g5 | A10G | 24 GB | 600 GB/s | 31 TF | ~0.5 TF (1/64) |
| g6e | L40S | 48 GB | 864 GB/s | 92 TF | ~1.4 TF (1/64) |

Both GPUs run FP64 at 1/64 of the FP32 rate, which drives the mixed-precision design below.

## Solver strategy: mixed-precision defect correction

- **Outer loop:** FP64 residual R(Q).
- **Inner loop:** FP32 linear relaxation on an approximate first-order Jacobian.
- The precision of the inner solve only changes how fast the solver converges. The converged state
  satisfies R₆₄(Q) = 0. SU2 and FUN3D use the same mixed-precision idea.

**Data types**

| Data | Warp type | Frequency |
|---|---|---|
| Grid metrics (volumes, face vectors, wall distance) | differenced in FP64, stored in residual precision | once at setup |
| Conservative state Q and residual R(Q) | `wp.float64` | once per outer step |
| Line blocks L, D, U | `wp.types.matrix((5, 5), wp.float32)` | every 1–5 outer steps |
| Correction δQ and RHS | `wp.types.vector(5, wp.float32)` | 5–10 inner sweeps |
| SA working variable ν̃ | `wp.float64` state, FP32 scalar line solve | segregated, right after the mean flow |

**Design rules**

1. **Smoother: wall-normal line-implicit, not point relaxation.**
   - On y⁺ ≈ 1 grids at Re 10⁶–10⁷, cell aspect ratios reach 10³–10⁵. Point Jacobi or point
     red-black Gauss–Seidel stall in the boundary layer.
   - Solver: line Gauss–Seidel with **red-black (zebra) line coloring**, colored by the parity of the
     parent-block (i, k) indices.
   - One thread per line runs a 5×5 block-tridiagonal Thomas solve in FP32.
   - Off-line neighbours are applied matrix-free (Rusanov-split A±·δQ).
   - Line-Jacobi is implemented first and stays available as an option.
2. **Metrics are differenced in FP64 and stored in the residual precision.**
   - `compute_metrics` ([grid/metrics.py](../src/wmles_wrap/grid/metrics.py)) currently casts the
     coordinates to `prec` before differencing.
   - In FP32, a 10⁻⁶c first cell with coordinates up to 30c loses all significant digits.
3. **The partitioner keeps lines intact.**
   - `split_blocks` ([grid/partition.py](../src/wmles_wrap/grid/partition.py)) must not bisect a
     parent block along its line axis.
   - This keeps convergence and results independent of the partitioning, as the existing tests
     require.
4. **Residual precision is a switch, and its cost is measured first.**
   - Estimated budget per outer step: 1 FP64 residual (~3–8 units), Jacobian assembly every 1–5
     steps (~2 units), and 5–10 sweeps (~1 unit each). The FP64 residual is then ~20–40 % of the
     cost.
   - Measure with `verification/benchmark.py --precision float64` vs `--precision float32` on g5
     and g6e.
   - Option `rans.residual_precision: float64 | float32`. FP32 residuals stall near 10⁻⁶ relative.
5. **Start simple:** plain defect correction with a switched-evolution-relaxation (SER) CFL ramp.
   Accelerators come in R4.

## Wall distance on the GPU

`scipy.spatial.cKDTree` runs only on the CPU, and the nearest-centroid distance is inexact near the
wall. Use Warp's BVH instead:

1. Collect all wall faces of all parent blocks (BC `wall` / `wall_modeled`) and split each quad
   into two triangles.
   - Build one `wp.Mesh` per device; every device holds the full wall surface.
2. A kernel per sub-block calls `wp.mesh_query_point_no_sign` at each cell centroid and keeps the
   index of the closest face.
   - The BVH works in FP32 (`wp.Mesh` points are `vec3f`), so it is used only to *find* the face.
3. Recompute the point-to-triangle distance to that face in FP64, from the FP64 coordinates.
   - Near-ties between faces only occur when the candidate distances are nearly equal, so they do
     not affect the result.
4. Run once at setup and store d in the residual precision.
   - Checked on the CPU build of Warp 1.19: `wp.Mesh` and `wp.mesh_query_point_no_sign` work.
   - The FP32 distance at d = 10⁻⁶ with O(1) coordinates came out as 1.000444×10⁻⁶ (a 4×10⁻⁴
     error), so the FP64 recompute in step 3 is needed.
   - Tests: exact analytic distance (flat plate, cylinder) and a brute-force numpy reference on a
     small grid.

## Stages

### R0: prerequisites

- FP64 metric differencing.
- A line-axis constraint in the partitioner.
- Mixed-precision kernel sets, using the existing `kernel_factory` cache in
  [precision.py](../src/wmles_wrap/precision.py).
- GPU wall distance (above).
- CGNS reader `io/cgns.py`:
  - Built on `h5py` and aimed at structured CGNS/HDF5 exports from ICEM and ANSA.
  - Reads structured zones, `ZoneGridConnectivity` 1-to-1 and `ZoneBC`.
  - Parent blocks are **split at sub-face boundaries**, so every interface becomes full-face and
    the existing connectivity code stays unchanged.
- TMR grid input (needed by R2/R3):
  - Formatted (ASCII) Plot3D in [io/plot3d.py](../src/wmles_wrap/io/plot3d.py), which reads only
    unformatted/raw binary today.
  - A NASA neutral map file (`.nmf`) reader that maps the TMR BC names (`viscous_solid`,
    `symmetry_*`, `subsonic_inflow_pt`, `back_pressure`, `farfield_riem`) and the 1-to-1 interfaces
    onto our BCs. NMF face numbers and sub-face ranges are converted to our faces, and partial
    faces are split as in the CGNS reader.
  - The TMR files are already 3D, two planes thick in i, so they need no `grid.extrude`. The i
    faces become periodic or symmetry.

### R1: pseudo-time implicit core (`numerics/implicit.py`)

- Local time stepping and an SER CFL ramp.
- First-order Rusanov/Roe Jacobians in FP32.
- Line-Jacobi, then zebra line Gauss–Seidel.
- δQ halo exchange between sweeps, using the existing halo maps with an FP32 vec5 array.
- Verify:
  - Laminar steady cases converge to machine zero.
  - The converged result matches the explicit solver's steady state.

### R2: SA model (`physics/sa.py` plus kernels)

- Negative-SA (Allmaras, Johnson & Spalart 2012) for robustness.
- First-order upwind convection for ν̃.
- Segregated scalar tridiagonal line solve for ν̃, run right after the mean flow.
- μ_t feeds the existing viscous flux.
- Verify:
  - NASA TMR SA manufactured solution (MMS).
  - TMR 2D zero-pressure-gradient flat plate (2DZP): Cf and grid convergence (see R3).

### R3: TMR verification

These three NASA TMR verification cases are the turbulence-model verification set. Each is
compared with the CFL3D/FUN3D SA results on the same grid family, using the grid-convergence plots
against h = 1/√N.

- **2DZP, zero-pressure-gradient flat plate**
  ([TMR page](https://tmbwg.github.io/turbmodels/flatplate.html), `cases/tmr/2DZP/`)
  - M = 0.2, Re = 5×10⁶ per unit length. Plate 0 ≤ x ≤ 2 behind a symmetry run-in.
  - Compare C_f at x = 0.97 (`cf_convergence.dat`) and plate drag C_D (`drag_convergence.dat`).
  - Also C_f(x) and the u⁺–y⁺ profile on the finest grid.
- **2DCJ, coflowing jet** (formerly "2D planar shear";
  [TMR page](https://tmbwg.github.io/turbmodels/shear.html), `cases/tmr/2DCJ/`)
  - A free jet from between two thin plates (one wall at y = 0.5, −10 < x < 0, centerline
    symmetry). Inner stream at M ≈ 0.5, outer stream at M ≈ 0.25, Re = 5×10⁴ per unit length.
  - It is a 3-block grid, so it also tests the multiblock interfaces. The free shear layer tests
    SA away from walls. The farfield ν̃ is set to at least 3ν_∞.
  - Wall distance must be computed to the true wall. This tests the BVH wall distance near the
    plate trailing edge.
  - Compare plate C_D and u at x = 2.71623, 29.2468 and 95.501 (`convergence.dat`), and u along
    y = 0 (`uvel_y_0.dat`).
- **2DB, bump-in-channel**
  ([TMR page](https://tmbwg.github.io/turbmodels/bump.html), `cases/tmr/2DB/`)
  - M = 0.2, Re = 3×10⁶ per unit length. Curved wall with a pressure gradient.
  - Compare C_f at x = 0.75 (`cf_convergence_sa.dat`), plus Cp and C_f along the bump.
- Next, outside the TMR verification set: the NACA 0012 at α = 0°, 10° and 15°.

### R4: acceleration

- FGMRES with the line Gauss–Seidel preconditioner. The outer residual is FP64; the Krylov
  vectors are FP32 with a small restart length, to fit 24 GB.
- And/or geometric FAS multigrid on the structured blocks.

### R5: production steady RANS on the transonic NASA CRM

- **First configuration: DPW-5 wing-body.**
  - Conditions: M = 0.85, Re = 5×10⁶ based on c_ref = 7.005 m (275.8 in), T = 310.93 K (100 °F).
  - Target C_L = 0.50.
  - Grids: the DPW-5 common multiblock structured family (tiny to ultra-fine, about 0.6 M to
    about 150 M cells).
- **Second configuration: DPW-6 wing-body** with the measured aeroelastic wing shape at each α.
  The flight shape comes from the R8 step-1 tool or from the supplied DPW-6 grids.
  - α sweep up to buffet onset.
  - This is the bridge to R6/R7 and to the Phase 4 buffet work in [plan.md](plan.md).
- **Reference data:**
  - SA results from CFL3D, FUN3D and OVERFLOW in the DPW-5/6 summary papers.
  - NASA NTF and Ames 11-ft experiments.
- **Comparisons:**
  - C_L, C_D and C_M.
  - Grid convergence of C_D (in counts) against 1/N^(2/3).
  - Cp at the wing stations.
  - Size of the side-of-body separation.
- **Features the CRM needs beyond R0–R4:**
  - Partial-face interfaces, handled by the block splitting in R0.
  - Degenerate (collapsed) faces and edges at the wing tip and at the fuselage nose and tail. This
    needs zero-area-face guards in the metrics, gradients and line solver.
  - A symmetry-plane BC: the existing `slip` BC with mirror ghosts, checked on this geometry.
  - A C_L driver that adjusts α to reach the target C_L.
  - Force and moment breakdown by component (wing, fuselage).
- **Memory:** about 700 B/cell, so about 30 M cells per A10G and about 60 M per L40S.
  - FP64 Q, R, W and gradients: about 260 B.
  - FP32 L/D/U line blocks: 300 B.
  - δQ, RHS, metrics and SA: about 150 B.
  - One L40S holds the DPW fine grid; 8×L40S hold the extra-fine and ultra-fine grids.
- **Grids:** downloaded by the user from the DPW site into `cases/crm/` (git-ignored).

### R6: dual-time stepping (URANS)

- BDF2 in physical time, with pseudo-time inner iterations using the R1/R4 solver.
- Time discretization:
  - The physical-time source (3Qⁿ⁺¹ − 4Qⁿ + Qⁿ⁻¹)/(2Δt) is added to the FP64 residual.
  - 3V/(2Δt) is added to the diagonal blocks.
  - BDF1 start-up. ν̃ also uses BDF2.
- Store Qⁿ and Qⁿ⁻¹ in FP64 (+80 B/cell).
- Inner iterations:
  - Stop on a residual drop (e.g. 2–3 orders) or a maximum subiteration count.
  - Limit the pseudo-time local steps relative to Δt for stability.
- Statistics, probes and the force monitor reuse the LES infrastructure (`k_stats`, `forces.dat`).
- Verify:
  - Temporal order with an isentropic convected vortex.
  - Laminar cylinder shedding at Re = 100: St ≈ 0.165, C_D, C_L amplitude.
  - URANS-SA OAT15A buffet onset and frequency (links to Phase 4).

### R7: DDES

- SA-DDES (Spalart et al. 2006), with an IDDES option (Shur et al. 2008).
- Length scale: d̃ = d − f_d·max(0, d − C_DES·Δ), with the shielding function f_d.
- Δ options:
  - Δ_max.
  - The shear-layer-adapted Δ_SLA (Shur et al. 2015), which shortens the grey area.
- Numerics in LES regions:
  - Reuse the existing KEEP-PE flux with sensor-weighted Roe dissipation.
  - A hybrid blending (Travin et al. 2004) raises the upwind weight in RANS and irrotational
    regions.
- Runs with R6 dual-time. The implicit inner solve lets the physical Δt be set by the LES region,
  not by the acoustic CFL of the y⁺ ≈ 1 cells.
- Verify:
  - C_DES calibration on decaying isotropic turbulence (reuse `verification/isotropic.py`).
  - NASA TMR 2D wall-mounted hump: separation, reattachment, Cp, Cf.
  - OAT15A buffet with DDES: experimental frequency and pressure RMS, and comparison with the WMLES
    results from Phase 4.

### R8: mesh deformation

**Step 1: flight mesh from the jig mesh (first deliverable)**
- Inputs, in a YAML `deform:` block:
  - The elastic axis as a polyline of points from root to tip.
  - Tables of bending deflection w(η) and twist θ(η) about the axis (nose-up positive). An in-plane
    (fore-aft) deflection is optional. η is the normalized distance along the axis.
  - The surface patches that move (the wing BC faces) and the ones that stay fixed (fuselage,
    symmetry plane, far field).
- Surface motion:
  - Project each wing surface node onto the elastic axis to get η.
  - Rotate the section about the local axis by θ(η).
  - Rotate it by the bending slope dw/dη, so sections stay normal to the bent axis.
  - Translate it by w(η).
  - The tables are cubic splines. θ = w = 0 at the wing root/junction, so the displacement blends
    to zero there.
- Volume:
  - Propagate the surface displacement with inverse-distance weighting (IDW; Luke, Collins &
    Blades 2012). IDW is linear in the surface displacement.
  - For larger twist, interpolate the rotation as well as the translation (same paper).
  - Symmetry-plane nodes keep their normal coordinate. Far-field nodes stay fixed.
- Cost: IDW is O(N_vol × N_surf). Use a reduced, greedy-selected surface point set and one Warp
  kernel per sub-block, with FP64 coordinates.
- Tool:
  - A CLI command `python -m wmles_wrap deform case.yaml` writes the flight-shape Plot3D/CGNS grid.
    It is an offline step, like `extrude`.
  - Alternatively, `grid.deform` applies it at setup, before the metrics.
- Quality checks:
  - Positive cell volumes and the minimum Jacobian ratio, from `compute_metrics`
    ([grid/metrics.py](../src/wmles_wrap/grid/metrics.py)).
  - Interfaces stay point-matched, so connectivity detection still succeeds on the deformed grid.
- Verify:
  - A rigid rotation and translation of the whole grid is reproduced exactly.
  - Analytic twist and bending on a straight rectangular wing: the sections match the prescribed
    θ and w.
  - CRM: deform the DPW jig grid with the DPW-6 measured twist and bending at each α, and compare
    the surface with the deformed geometry supplied by DPW-6.

**Step 2: general surface → volume operator for R9**
- The same IDW, as a Warp function of the surface displacement.
- Its transpose comes from `wp.Tape` or is written explicitly, and is checked with a dot-product test.
- Later, an FFD or Hicks–Henne surface parametrization feeds it.

**Later (optional):** static aeroelastic coupling, where twist and bending come from a beam model
loaded by the RANS surface forces.

### R9: discrete adjoint and shape gradients (target)

**Assessment**
- Building gradient-based optimization now would be too ambitious.
- Making it a design target is realistic and cheap, *if* the rules below hold from R0 onward.
- Retrofitting them later would mean rewriting the residual.

**Method: discrete adjoint, not a taped solve.** The adjoint of the steady state uses the same
fixed-point structure as the primal, in mixed precision:

  ψ ← ψ − P⁻ᵀ (∂R/∂Qᵀ ψ − ∂J/∂Qᵀ),   then   dJ/dX = ∂J/∂X − ψᵀ ∂R/∂X

- The vector–Jacobian products ∂R/∂Qᵀψ and ∂R/∂Xᵀψ come from Warp's built-in reverse mode
  (`wp.Tape` over the residual kernels only, for one residual evaluation). They are computed in FP64.
- P⁻ᵀ is the transposed FP32 line solver, applied to the same approximate Jacobian.
- The adjoint therefore converges at the primal rate and needs no hand-derived Jacobian.

**Design rules from R0, enforced by tests**
1. **The residual is a pure function R(Q, X).**
   - The pipeline Q → W → ghosts → gradients → fluxes → R writes only to fresh arrays.
   - No kernel overwrites an array that an earlier kernel in the same residual has read.
   - Updates to the state stay outside the residual, as `k_rk` does today.
   - A test checks this with Warp's `verify_autograd_array_access`.
2. **Metrics are a differentiable Warp function of the node coordinates.**
   - No host-side numpy steps between X and R.
   - Mesh deformation (surface → volume) is R8. It is linear (inverse-distance weighting), so it
     can be transposed.
3. **Wall distance:** the BVH only selects the closest face. The FP64 point-to-triangle kernel is
   differentiable, so frozen-face d(X) has a gradient.
4. **No data-dependent iteration inside the residual.**
   - The equilibrium wall-model fixed point is LES-only.
   - RANS uses only smooth or piecewise expressions (min, max and abs are allowed).
   - Use smooth variants where it matters, such as the entropy fix and the sensor.
5. **No atomics in the forward residual.** This already holds (deterministic face differencing).
   Warp's adjoint of the gather kernels scatters with atomic adds, which is accepted.
6. **Building blocks must be transposable.**
   - Halo maps are explicit index lists, so their transpose is a scatter-add from ghost back to
     donor (add `HaloExchange.exchange_adjoint`).
   - BC ghost kernels get their vector–Jacobian product from the tape.
   - The block-tridiagonal solver supports the transpose (solve with Dᵀ and swap Lᵀ/Uᵀ).
7. **Objectives are kernels.** C_L, C_D and C_M are already integrated on the device
   (`k_wall_sample`) and stay as differentiable kernels.
8. **Warp backward setting.**
   - `wp.config.enable_backward = False` is currently set globally in
     [parallel/devices.py](../src/wmles_wrap/parallel/devices.py).
   - It becomes a per-module setting: backward code is generated only for the residual module, and
     only when the adjoint is requested, so primal compile times are unchanged.
9. **Test harness from R1:**
   - A dot-product test ⟨v, J u⟩ = ⟨Jᵀv, u⟩ on small grids in FP64.
   - A comparison of tape vector–Jacobian products with central finite differences.

**Scope of R9 (after R5 and R8)**
- The adjoint solver.
- dC_D/dX_surface checked against finite differences on the NACA 0012 and on the CRM wing.
- Later: FFD or Hicks–Henne parametrization and coupling to an optimizer (SciPy SLSQP).
- Adjoint memory is about 2–3× the residual working set, because the tape stores the residual
  intermediates. That roughly halves the maximum grid size per GPU.

### R10: chimera grids

- Overlapping structured component grids (wing, fuselage, horizontal tail) replace a single
  point-matched multiblock grid, so the tail and other components can be meshed independently.
- Hole cutting against the component surfaces reuses the `wp.Mesh` BVH from the wall-distance
  section. It is followed by a fringe/donor search and trilinear interpolation of the fringe cells.
- Fringe interpolation fits the halo structure. It uses explicit donor-stencil lists that are
  exchanged like the halo maps, so it stays transposable for the adjoint and works across devices.
- Implicit solver: fringe cells are frozen (Dirichlet) in the line solves, and lines stop at
  blanked cells.
- Wall distance and forces must handle overlapping surfaces. A zipper grid or weighted surface
  integration keeps overlapped walls from being counted twice.
- Verify:
  - Overset vs point-matched results on the same 2D case (e.g. the NACA 0012 with an overset
    background grid).
  - Then the CRM wing-body-tail (DPW-4 WBT0 configuration).

## Test grids

The NASA Turbulence Modeling Resource (TMR) and DPW grids are downloaded by the user. The TMR grids
are formatted Plot3D, 3D with two planes in i (one cell thick), plus a neutral map file (`.nmf`)
with the BCs. The same families are also there as structured CGNS (`*.cgns`), which serve as the
test input for the R0 CGNS reader. Grid files (`*.p3dfmt`, `*.cgns`) are git-ignored. The `.nmf` files and the SA reference data
(`*.dat`) are committed.

| Case | Directory | Grid sizes (downloaded) | Reference data |
|---|---|---|---|
| 2DZP flat plate | `cases/tmr/2DZP/` | 35×25, 69×49, 137×97, 273×193 (545×385 not downloaded) | C_f at x = 0.97, C_D |
| 2DCJ coflowing jet | `cases/tmr/2DCJ/` | 3 blocks, 4 levels: 9×17 ×2 + 33×33 up to 65×129 ×2 + 257×257 (finest 129×257 ×2 + 513×513 not downloaded) | Plate C_D, u at three x-stations, u(x) at y = 0 |
| 2DB bump-in-channel | `cases/tmr/2DB/` | 89×41, 177×81, 353×161 (705×321, 1409×641 not downloaded) | C_f at x = 0.75; Cp, C_f on the bump |
| NACA 0012 | `cases/tmr/naca0012/` | 113×33, 225×65, 449×129, 897×257 (1793×513 optional) | C_L, C_D, Cp at α = 0°, 10°, 15° |
| CRM wing-body | `cases/crm/` | DPW-5 common structured family | DPW-5/6 SA results, NTF/Ames data |

Also useful:
- The TMR SA MMS (R2).
- The TMR 2D wall-mounted hump grids and experimental data (R7).
