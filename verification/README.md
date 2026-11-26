# Verification

Scripts reproduce each study; results below are from the macOS CPU build (float64).
Run from the repository root with `conda run -n pyvista env PYTHONPATH=src python verification/<script>.py`.

## Phase 1: code verification

| Study | Script | Result |
|---|---|---|
| Free-stream preservation (curvilinear, reoriented 2-block grid, 3 partitions, far field / slip, all dissipation modes) | `tests/test_numerics.py` | max deviation ≤ 1.2e-14 |
| MMS, Navier–Stokes (Sutherland), periodic + block interface + Dirichlet, `mms_order.py` | `mms_order.py` | L2 order 1.94–2.10 (all variables), n = 8/16/32 |
| MMS, Euler | `mms_order.py --euler` | L2 order 1.93–2.05 |
| MMS, NS + Roe + `ducros_pressure` sensor | `mms_order.py --dissipation roe` | L2 order 1.89–2.14 (plain `ducros` sensor: ~1.0) |
| Sod / Lax / 1000:1 shock tubes, n = 200 | `shock_tube.py --case ... --plot` | ρ L1 rel. error 1.08e-2 / 3.67e-2 / 1.21e-1; plots in `results/` |
| Couette with viscous heating (isothermal walls, moving wall) | `couette.py` | u exact to 1e-14; T error order 2.02, 2.00 |
| Inviscid Taylor–Green, KEEP-PE conservation | `tgv.py` | KE exchange ∝ M² (1.5e-3, 3.7e-4, 9.1e-5 for M = 0.2, 0.1, 0.05), no drift; mass/momentum/energy conserved to 1e-14; Roe + sensor changes KE by < 1e-6 |
| Decomposition consistency | `tests/test_numerics.py::test_partition_independence_bitwise` | bitwise identical for 1/3/5 partitions and staged (cross-device) halos |

Notes
- The default shock sensor is Ducros × Jameson-type pressure/density second-difference sensor, compression only, with a regularized Ducros (→ 1 where there are no velocity gradients). Plain Ducros is O(1) in irrotational smooth flow and makes the scheme first order there.
- Contact discontinuities receive dissipation only through the density term of the sensor; small oscillations remain near contacts, as expected from a minimally dissipative scheme.

Not yet done in Phase 1: isentropic vortex long-time transport, laminar compressible flat plate, CPU-vs-GPU comparison (needs the g6e node).

## Phase 2: SGS and compressible turbulence

| Study | Script | Result |
|---|---|---|
| SGS kernels (Vreman, WALE, Smagorinsky) vs numpy reference, random gradients | `tests/test_sgs.py` | agreement to 1e-12; Vreman/WALE vanish in pure shear |
| Taylor–Green vortex, Re = 1600, M = 0.1, Roe + `ducros_pressure` | `tgv.py --re 1600`, `plot_tgv.py` | see table below and `results/tgv_re1600_dissipation.png` |
| Decaying compressible isotropic turbulence, Mt = 0.6, Re_λ = 100 (Johnsen et al. 2010 setup) | `isotropic.py --n 32/64` | stable with shocklets; k/k0 at t/τ = 4: 0.289 (32³), 0.289 (64³); T_rms 2.9e-2 / 3.1e-2; sensor-active cells 14% → 6% with refinement |
| Wall-resolved channel Re_τ = 180 (pipeline) | `cases/channel_retau180`, `channel_profiles.py` | 2000-step CPU smoke run stable; u_τ from wall shear (0.08643) and from mass-flux forcing (0.08641) agree, i.e. the global momentum balance closes. Statistically converged profiles need a GPU run (~3×10⁵ steps) |
| Multi-GPU throughput / scaling | `benchmark.py` | to be run on the g6e.48xlarge (8× L40S) |

TGV Re = 1600 peak dissipation rate ε = −dk/dt (DNS: 0.0127 at t ≈ 9.0, van Rees et al. 2011):

| grid | no SGS | Vreman | WALE |
|---|---|---|---|
| 32³ | 0.0094 at t = 10.75 | 0.0116 at t = 6.45 | 0.0123 at t = 7.25 |
| 64³ | 0.0110 at t = 8.45 | 0.0104 at t = 7.55 | 0.0106 at t = 7.95 |

At t = 0 the no-SGS run reproduces the laminar dissipation ν⟨|ω|²⟩ = 0.75/1600 = 4.7e-4. Vreman and WALE add about 7e-4 in the laminar/transitional phase (neither vanishes for the 3D initial TGV field), which moves the peak earlier on coarse grids. This SGS contribution at t = 0 drops from 7.4e-4 (32³) to 1.9e-4 (64³), the expected Δ² scaling.

At 64³ the curves move toward the DNS, but none has converged: all peak about 15% low and about 1 time unit early. A second-order scheme needs about 128³ for this case (GPU run). In this low-Re transitional flow the no-SGS run (KEEP-PE with the sensor essentially inactive) is closest to the DNS at 64³, because Vreman and WALE over-dissipate during transition (shoulder near t ≈ 6). This is a property of the case, not of wall-bounded high-Re WMLES, where the SGS model is still needed.

## Phase 3: wall models

| Study | Script / test | Result |
|---|---|---|
| Warp wall-model kernel vs numpy reference (equilibrium ODE and algebraic; isothermal/adiabatic; both wall sides; exchange by cell index or distance) | `tests/test_wallmodel.py` | τ_w and q_w agree to 1e-10 |
| A priori, Lee & Moser Re_τ = 5200: DNS mean velocity at y = h → model u_τ (DNS u_τ/U_b = 0.04149) | `tests/test_wallmodel.py` | equilibrium: +0.01% (h = 0.05), +0.4% (0.1), +0.8% (0.2), +1.6% (0.3); algebraic (Reichardt): −3.0%, −2.5%, −1.9%, −1.6% |
| Fixed-point convergence (equilibrium model) | — | undamped: rel. error 2e-3 after 4 iterations, 1e-5 after 10 (cold start, 30 iterations by default; 4 with warm start) |
| Momentum balance in the solver: mass-flux forcing = modeled wall traction / volume | `tests/test_wallmodel.py::test_wall_traction_balances_forcing` | equal to 1e-10 for both models |
| WMLES channel Re_τ = 5200, UMD grids | `cases/channel/case.yaml`, `channel_profiles.py --utau forcing --lm ...` | ar2 grid 64×32×48 cells (extruded to 4.8h), M = 0.2, Re_b = 125 000, Vreman, exchange at cell 3 (h ≈ 0.067, h⁺ ≈ 350), isothermal walls, 60 000 steps (t ≈ 169 h/U_b), statistics over t ≈ 70–169: **equilibrium ODE u_τ/U_b = 0.0407 (−1.9%, Re_τ 5075)**; algebraic 0.0397 (−4.3%, Re_τ 4957). Both show the log-layer mismatch typical of coarse WMLES (u⁺ ≈ 3 low at the first cell, slope too steep) and an overpredicted u_rms⁺ near the wall; the outer region (y⁺ > 1000) matches DNS. Last three time-window averages of u_τ agree to ±0.5%. Plots: `results/channel_re5200_wm_compare_ar2.png` (+ per-model `.png`/`.txt`) |
| WMLES channel Re_τ = 5200, GPU sweep (1× A10G, float32, 200 000 steps, statistics from step 50 000) | `channel_sweep_1gpu.sh` | ar2 equilibrium cell 3: −2.2% (float64 CPU: −1.9%, so float32/GPU agrees). ar1 equilibrium, exchange cell 1/2/3/4/5 (h = 0.013/0.039/0.067/0.098/0.134, h⁺ = 65/200/346/509/697): **−15.9% / −8.0% / −3.9% / −1.4% / −0.04%**; ar1 algebraic cell 3: −6.1%. The exchange location dominates the error (log-layer mismatch); the finer streamwise grid costs about 2 points for both models at cell 3. About 70 Mcell-steps/s on the ar1 grid. Plot: `results/channel_re5200_exchange_sensitivity.png` |
| Wall-face diagnostics: force monitor, mean surface samples (p, τ_w, q) assembled across partitions, time-based sampling, statistics and wall-model state carried through a restart onto a different partitioning | `tests/test_wall_stats.py` | force equals a host recomputation to 1e-12; surface means equal the host averages; the restarted run reproduces the uninterrupted statistics to 1e-12 |
| A-airfoil near stall (α = 13.3°, Re_c = 1e7, M = 0.15), reference WRLES Tamaki & Kawai (AIAA J. 2023) | `cases/a_airfoil/case.yaml`, `cases/a_airfoil/coarsen.py`, `airfoil_surface.py` | set up. CPU smoke test on a 4×2×9-coarsened grid (516 k cells, 6 partitions): connectivity (wake cut, 3 block joins, spanwise periodicity) correct, GCL 1e-16, stable for 3000 steps (0.1 CTU) of the impulsive start, residuals down about 20×. Production run (36.7 M cells, 30 CTU, averaging over the last 10) pending on the GPU node |

## RANS-SA (docs/plan_rans.md stages R0–R2)

| Study | Script / test | Result |
|---|---|---|
| Grid input: TMR Plot3D+NMF vs CGNS (ADF), patch splitting, zero-thickness plate | `tests/test_grid_import.py` | identical blocks, cuts, interfaces and walls for 2DZP/2DB/2DCJ |
| Wall distance: analytic plate/cylinder, brute force, ridge near-ties | `tests/test_wall_distance.py` | ≤ 1e-12 relative (float64); FP32 BVH alone up to 857 % off near a ridge, 7e-9 with the FP64 re-check |
| Implicit core: Couette to machine zero, approximate Jacobian vs finite differences | `tests/test_implicit.py` | 1e-12 in < 60 iterations, matches the explicit steady state; line blocks within 6–8 % of the FD Jacobian |
| Steady MMS, Roe-MUSCL + negative SA (μ_t in the mean flow, analytic wall distance), reoriented 2-block grid, 3 partitions | `sa_mms.py` | L2 order, n = 8/16/32: mean flow 2.06–2.50, ν̃ 0.95–0.96 (first-order upwind convection, as designed); each grid converged in ≤ 91 iterations |
| Laminar flat plate, Re = 1e5 per unit length, TMR 2DZP grids | `tests/test_sa.py::test_blasius` | C_f/C_f,Blasius within 2.6 % (69×49) and 1.3 % (137×97) for 0.1 < x < 1.6; converged to 1e-11 in 164–214 iterations |
| TMR 2DZP flat plate, SA, M = 0.2, Re = 5e6 | `sa_flatplate.py` | see below |

TMR 2DZP with SA (C_f from the first wall cell, C_D = ½∫₀² C_f dx), each grid converged to a 1e-10 residual drop:

| N cells | iterations | C_f(0.97) | CFL3D | FUN3D | C_D | CFL3D | FUN3D |
|---|---|---|---|---|---|---|---|
| 816 | 503 | 2.7846e-3 | 2.7957e-3 | 2.6674e-3 | 2.9550e-3 | 2.9544e-3 | 2.7739e-3 |
| 3 264 | 219 | 2.7372e-3 | 2.7284e-3 | 2.6950e-3 | 2.9033e-3 | 2.8844e-3 | 2.8226e-3 |
| 13 056 | 183 | 2.7258e-3 | 2.7112e-3 | 2.7022e-3 | 2.8875e-3 | 2.8662e-3 | 2.8400e-3 |
| 52 224 | 167 | 2.7236e-3 | 2.7067e-3 | 2.7045e-3 | 2.8823e-3 | 2.8613e-3 | 2.8479e-3 |

Second-order grid convergence, to about 2.723e-3 (C_f) and 2.880e-3 (C_D): 0.65–0.85 % above the TMR
fine-grid values (2.7055e-3, 2.856e-3), where CFL3D and FUN3D agree to 0.01 %. Open for R3: the TMR
total-pressure inflow and back-pressure outflow (here simplified inflow/outflow BCs) and the wall-shear
evaluation.
