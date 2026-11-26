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
| R0 | Prerequisites: grid input (CGNS, NMF, formatted Plot3D), FP64 metrics, GPU wall distance, line-aware partitioner, mixed-precision kernels | done (branch `rans-r0`) |
| R1 | Pseudo-time implicit core (mixed-precision defect correction, line relaxation) | done (laminar; branch `rans-r0`) |
| R2 | Dissipative steady residual, then the SA model (negative-SA) | done (branch `rans-r2`) |
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
   - In FP32, a 10⁻⁶c first cell with coordinates up to 30c loses all significant digits.
   - Done: `compute_metrics` ([grid/metrics.py](../src/wmles_wrap/grid/metrics.py)) differences
     in FP64 and casts the results; the GCL diagnostic is evaluated on the stored face vectors.
   - Still open for FP32 residuals: the flux face-gradient correction and the wall-model exchange
     height use differences of the stored centroids `xc`, so they lose the same digits. If
     `rans.residual_precision: float32` is pursued, store FP64-differenced cell-to-cell
     displacement vectors per face (and the wall distance from R0) instead.
3. **The partitioner keeps lines intact.**
   - `split_blocks` ([grid/partition.py](../src/wmles_wrap/grid/partition.py)) must not bisect a
     parent block along its line axis.
   - This keeps convergence and results independent of the partitioning, as the existing tests
     require.
   - Done: `parallel.keep_lines: true` (default off, so LES partitioning is unchanged). The line
     axis of a parent block is its wall-normal axis, or, without walls (wake, outer blocks), the
     axis of the high-aspect-ratio cells; one-cell-thick axes are never line axes. Lines still
     end at parent-block boundaries, including the cuts made by the patch splitting (e.g. the
     2DCJ downstream block at k = 16); those do not depend on the partitioning.
4. **Residual precision is a switch, and its cost is measured first.**
   - Estimated budget per outer step: 1 FP64 residual (~3–8 units), Jacobian assembly every 1–5
     steps (~2 units), and 5–10 sweeps (~1 unit each). The FP64 residual is then ~20–40 % of the
     cost.
   - Measure with `verification/benchmark.py --precision float64` vs `--precision float32` on g5
     and g6e.
   - Option `rans.residual_precision: float64 | float32`. FP32 residuals stall near 10⁻⁶ relative.
5. **Start simple:** plain defect correction with a switched-evolution-relaxation (SER) CFL ramp.
   Accelerators come in R4.

## Wall distance on the GPU (done)

`scipy.spatial.cKDTree` runs only on the CPU, and the nearest-centroid distance is inexact near the
wall. [grid/wall_distance.py](../src/wmles_wrap/grid/wall_distance.py) uses Warp's BVH instead
(`Solver.wall_distance()`, computed on first use; output field `wall_distance`):

1. Collect all wall faces of all parent blocks (BC `wall` / `wall_modeled`) and split each quad
   into two triangles; zero-area triangles of collapsed faces are dropped.
   - One `wp.Mesh` per device; every device holds the full wall surface.
   - Coordinates are shifted by the centre of the wall surface before the FP32 cast.
2. `k_select`, one thread per padded cell: `wp.mesh_query_point_no_sign` at the FP32-rounded
   centroid gives a closest triangle. FP32 rounding can pick the wrong one near edges between
   faces at an angle, so a FP32 box query of half-size d + 4δ (δ bounds the FP32 rounding of
   the coordinates) re-checks the candidates in FP64. The loop is capped at 256 candidates; a
   partial check can only lower d.
   - The original assumption that near-ties do not matter was wrong: 30 units from the surface
     centre, points 10⁻⁷–10⁻⁵ above a sharp ridge get up to 857 % error from the FP32 selection
     alone, and 7×10⁻⁹ (FP64 rounding) with the re-check.
3. `k_distance`: FP64 point-to-triangle distance (Ericson) to the selected triangle, from FP64
   centroids and vertices, stored in the run precision. The selected triangle is kept, so d(X)
   with frozen faces is a separate differentiable kernel (adjoint design rule 3).
4. Tests ([tests/test_wall_distance.py](../tests/test_wall_distance.py)):
   - Exact analytic distances: flat plate with a run-in (float64: 10⁻¹³; float32 storage 30 units
     out with a 10⁻⁶ first cell: FP32 rounding only), faceted cylinder (10⁻¹²), TMR 2DZP grid.
   - A brute-force numpy reference on a curvilinear two-block grid, partitioning independence, the
     ridge near-tie case, no walls, collapsed faces.
   - 2DCJ: both sides of the zero-thickness plate match the analytic strip distance to 7×10⁻¹⁴.
   - Cost on the CPU build: 2.3 s for the finest 2DZP grid (1.08M padded cells), compile included.

## Stages

### R0: prerequisites

- **FP64 metric differencing (done):** face vectors, volumes and centroids are computed in FP64
  and stored in the run precision. On a 10⁻⁶ first cell 30 units from the origin, the FP32 volume
  error drops from 138 % (with zero volumes) to 4×10⁻⁸. Float64 runs are bitwise unchanged.
- **Line-axis constraint in the partitioner (done):** `line_axes` and `split_blocks(keep_axes=...)`
  in [grid/partition.py](../src/wmles_wrap/grid/partition.py), see design rule 3.
- **Mixed-precision kernel sets (done):** `MixedPrecision` / `get_mixed(residual, solve)` in
  [precision.py](../src/wmles_wrap/precision.py) (cached by `kernel_factory` like `Precision`;
  `Precision.mat55` is the 5×5 block type), and [numerics/mixed.py](../src/wmles_wrap/numerics/mixed.py):
  - `k_rhs` (rhs = −R, residual → solve precision) and `k_correct` (Q += δQ, solve → residual
    precision) on the interior cells.
  - `block_solve`, a 5×5 Gaussian elimination with partial pivoting: `wp.inverse` only goes up to
    4×4, and the block-tridiagonal Thomas solve of R1 needs 5×5 diagonal-block solves.
  - The `rans.residual_precision` option comes with the R1 solver.
- **GPU wall distance (done, above).**
- **Grid input with patch information (done, branch `rans-r0`):**
  - Structured CGNS reader [io/cgns.py](../src/wmles_wrap/io/cgns.py) for both storage formats:
    HDF5 through `h5py` (ICEM/ANSA/Pointwise exports), and ADF through a small pure-Python reader of
    the ADF on-disk layout (the TMR CGNS files are ADF, written in 2009). It reads structured zones,
    `GridCoordinates`, `ZoneBC` (vertex `PointRange`, family names) and `GridConnectivity1to1`.
    `PointList`, general `GridConnectivity_t` and links are rejected with a clear error.
  - NASA neutral map file reader [io/nmf.py](../src/wmles_wrap/io/nmf.py) (`grid.nmf`), including
    swapped and reversed 1-to-1 ranges.
  - Formatted (ASCII) Plot3D in [io/plot3d.py](../src/wmles_wrap/io/plot3d.py), detected
    automatically.
  - [grid/patches.py](../src/wmles_wrap/grid/patches.py) splits blocks at every patch boundary and
    carries the cuts through the 1-to-1 interfaces until no new cut appears. Every patch then
    covers whole faces, so the existing full-face connectivity code is unchanged; the imported
    1-to-1 list only checks the geometric matching.
  - BCs select faces by patch label (`boundaries[].patches`: NMF names; CGNS BC names, BC types and
    families). Faces with a BC patch are never matched geometrically except through periodicity,
    so the two coincident wall faces of the zero-thickness 2DCJ plate stay walls.
  - The BC kernel handles blocks thinner than the ghost layers (the one-cell-thick TMR grids).
  - Verified: the TMR Plot3D+NMF and CGNS versions of 2DZP, 2DB and 2DCJ give identical blocks,
    splits, interfaces and walls; a 2DCJ explicit run is bitwise partition-independent.
  - The TMR files are already 3D, two planes thick in i, so they need no `grid.extrude`; the i
    faces are symmetry (slip) or periodic.

### R1: pseudo-time implicit core (`numerics/implicit.py`) — done

`solver.mode: rans` with a `rans:` block (`turbulence: none` for now). Each outer iteration solves
one backward-Euler pseudo-time step, (V/Δτ − ∂R/∂Q) δQ = R, approximately; Q += α δQ.

- **Local time stepping and SER:** V/Δτ = λ/CFL from the explicit spectral radius;
  CFL = cfl_start·(res₀/res)^β up to cfl_max, rounded to 4 significant digits so the round-off of
  the global norm cannot make runs partitioning-dependent. A per-cell α ≤ 1 limits the relative
  density and pressure change (`relax_limit`).
- **Approximate Jacobian (FP32 by default, `rans.solve_precision`):**
  - Inviscid: ½A(Q_j, S) per face (exact Euler flux Jacobian); the cell's own central terms
    cancel by the GCL.
  - Dissipation: Rusanov or Roe (`rans.jacobian`; R2 made `auto` the default: Roe for a Roe residual), scaled by
    `rans.implicit_dissipation`.
  - Viscous: thin-layer, linear in V = (ρ, u, T): K_f = |S|²/V_f [μ(I + nnᵀ/3) on u, κ on T, work
    term], so K ∂V/∂U couples the energy equation to ρ and ρu through T. A scalar viscous
    coefficient on ρE was tested first: it misses dRes_E/dρ (about 3× the diagonal on Couette) and
    the iteration stalls.
  - Wall and slip faces: the linearized ghost state δQ_g = B δQ_i (isothermal ρ_g = p/(R T_g)
    included) and no dissipation, as in the residual. The operator then conserves mass like the
    residual; with dissipation or viscous mass terms at walls the corrections pumped mass along
    the residual's null space (Couette mass 1.0 → 1.045, then 12.9) and convergence stalled.
  - Farfield, inflow and outflow are explicit (δQ_ghost = 0).
  - Checked against finite-difference Jacobians of the residual (tests).
- **Line relaxation:** only the two line blocks are stored; off-line neighbours (and line
  neighbours across block interfaces) are matrix-free products with the latest δQ. One thread
  per line runs a 5×5 block Thomas solve (`block_inverse`). `zebra` (default) colours lines by
  the parity of the parent-block off-line indices; `line_jacobi` double-buffers. δQ halos are
  exchanged after every colour/sweep. The partitioner keeps lines whole automatically.
- **Results** (tests in [tests/test_implicit.py](../tests/test_implicit.py)):
  - Compressible Couette (isothermal walls, viscous heating): 10⁻¹² drop in < 60 iterations
    (32 with `implicit_dissipation: 0`); u matches the exact linear profile to 10⁻¹⁰; u, T and
    ρ/ρ̄ match the explicit solver's steady state.
  - Laminar flat plate (Re = 10⁴, TMR 2DZP 35×25 grid and a synthetic one), Rusanov residual and
    Jacobian: 10⁻¹² in 163 iterations (zebra), 294 (line-Jacobi), 221 on the 69×49 grid.
  - Bitwise independent of the partitioning; converged states independent of the smoother and
    of the solve precision (FP32 vs FP64) to 10⁻⁸.
  - Local pseudo-time stepping conserves Σ(V/Δτ)δρ, not the total mass: in closed domains
    (Couette) the converged mass level can differ slightly from the explicit one.
- **Findings that shape R2/R3:**
  - The convergence rate is set by how well the operator matches the residual, not by the
    sweeps or the CFL. With the LES residual (central KEEP-PE + sensor-weighted dissipation) the
    implicit dissipation that stability needs damps modes the residual does not, and steady
    convergence is slow (plate: 5×10⁻⁷ after 400 iterations). A Roe residual with the Rusanov
    operator converges at ~0.98 per iteration.
  - The Roe operator matched the Roe residual closely but limit-cycled near 10⁻⁷. Attributed
    here to the |u·n| kink; R2 found the cause in the far-field BC switching at u_n = 0.
  - Hence the dissipative, smooth steady residual added to R2 below.
- Not done yet: GPU timing and memory per cell, implicit far-field/inflow/outflow BCs.

### R2: dissipative steady residual, then the SA model (`numerics/sa.py`) — done

- **Steady residual** `numerics.flux: roe_muscl`: van Albada primitive MUSCL (smooth;
  `numerics.muscl_eps` relative to free-stream scales) and Roe flux; first-order mirrored states and
  no dissipation on wall/slip faces; first-order fallback if a reconstructed ρ or p is not
  positive. The LES residual (`keep_pe`) is bitwise unchanged. Optional Harten fix on the
  entropy/shear waves (`numerics.entropy_fix_linear`, default 0: 0.05 raised C_f by 5–20 %).
- **Negative SA** (Allmaras, Johnson & Spalart 2012, with f_t2, no trip; `rans.turbulence: sa`):
  ν̃ is a separate padded scalar (`Solver.turbulence.nut`) with its own ghosts (wall −ν̃, slip +ν̃,
  far field ν̃∞ inflow / extrapolation outflow) and halo exchange; μ_t = ρν̃f_v1 replaces the SGS
  viscosity in `Integrator.rhs`; first-order upwind convection, diffusion
  ∇·((ν + ν̃f_n + c_b2ν̃)∇ν̃) − c_b2ν̃∇²ν̃, vorticity and the GPU wall distance in the source.
  ν̃∞ = 3ν∞ (`rans.nut_inf_ratio`). Restart files and output fields `nut`, `mut` include it.
- **Segregated implicit step for ν̃** after the mean flow: scalar line solves (same zebra
  colouring), upwind convection, frozen-coefficient diffusion, and the positive part of the full
  d(D − P)/dν̃ by a float64 finite difference. The simplified 2c_w1f_wν̃/d² misses the f_w(r(ν̃))
  term (more than half near walls) and the iteration oscillated at high CFL.
- **Convergence findings** (each was a stall on the TMR plate; tests and README record them):
  - The far-field Riemann BC switched its entropy/tangential reference state at u_n = 0, which is
    exactly the top boundary of a flat plate. In `solver.mode: rans` the switch is blended
    smoothly (tanh over 0.05 a), for the mean flow and ν̃. This, not the |u·n| kink, caused the
    Roe-operator stalls recorded in R1.
  - Operator dissipation must match the residual's: the Roe operator now floors the acoustic
    waves at 0.2a (`rans.jacobian_entropy_fix`) and the entropy/shear waves at 0.05a
    (`..._linear`); a 0.2a floor on all waves over-damps the shear waves near walls and converges
    very slowly at high Re. `rans.jacobian: auto` picks Roe for a Roe residual.
  - Second-order residuals respond more strongly than the first-order operator in some modes;
    exact linear solves then overshoot. `rans.implicit_dissipation: 1.25` (default) fixes it.
  - Fine grids need more linear sweeps: adaptive, until max|δQ_new − δQ_old| ≤ 0.02 max|δQ|
    (`rans.sweep_tolerance`, max norms so the count is partition-independent), 4–32 sweeps.
- **Verification** ([verification/README.md](../verification/README.md)):
  - Steady MMS with SA (`verification/sa_mms.py`): mean flow order 2.1–2.5, ν̃ 0.96.
  - Laminar plate vs Blasius: 1.3 % on the 137×97 grid.
  - TMR 2DZP SA (`verification/sa_flatplate.py`): converged to 1e-10 on all four grids
    (503/219/183/167 iterations); C_f(0.97) 2.7846e-3 on 35×25 vs CFL3D 2.7957e-3 and FUN3D
    2.6674e-3; second-order grid convergence to ~2.723e-3, 0.65 % above the TMR fine-grid value.
    The TMR inflow/outflow BCs are open for R3.

### R3: TMR verification

First, from R2: the TMR total-pressure/total-temperature inflow and back-pressure outflow BCs
(the 2DZP results are 0.65 % above TMR with the simplified inflow/outflow), and a second-order
wall-shear evaluation.

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
