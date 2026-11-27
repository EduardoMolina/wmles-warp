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
| TMR coarse-grid regressions (2DCJ 1280 cells, 2DB 89×41, NACA 0012 II.7) | `tests/test_tmr.py` | 2DCJ centreline u within 0.0015 a_ref of CFL3D; 2DB C_f within 1.5 % of CFL3D |
| TMR open BCs: inflow_total (p0, T0, direction, R+), back pressure; linearized ghosts vs FD Jacobian; wall data | `tests/test_open_bcs.py` | conditions to 1e-12; boundary blocks within 6 % of FD (85 % with frozen ghosts) |
| TMR NACA 0012, α = 10°, SA, family II | `sa_naca0012.py` | see below |

All TMR runs use `numerics.flux: roe_muscl` with `numerics.limiter: none` (unlimited κ = 1/3 MUSCL, as
CFL3D runs these subsonic cases). The default van Albada limiter clips smooth peaks: on the NACA 0012
it gave +59 % pressure drag on family II.6 (C_Dp 0.0141 vs CFL3D 0.0089), on the TMR bump and plate it
mainly degraded the coarse grids.

TMR 2DZP with SA, TMR boundary conditions (inflow_total: p0/p∞ = 1.02828, T0/T∞ = 1.008; back
pressure p/p∞ = 1; Riemann far field), C_f from the two-cell wall derivative (`post/surface.py`),
C_D = ½∫₀² C_f dx, each grid converged to a 1e-10 residual drop (545×385 from the CGNS file):

| N cells | iterations | C_f(0.97) | CFL3D | FUN3D | C_D | CFL3D | FUN3D |
|---|---|---|---|---|---|---|---|
| 816 | 526 | 2.7719e-3 | 2.7957e-3 | 2.6674e-3 | 2.9343e-3 | 2.9544e-3 | 2.7739e-3 |
| 3 264 | 245 | 2.7214e-3 | 2.7284e-3 | 2.6950e-3 | 2.8800e-3 | 2.8844e-3 | 2.8226e-3 |
| 13 056 | 246 | 2.7088e-3 | 2.7112e-3 | 2.7022e-3 | 2.8653e-3 | 2.8662e-3 | 2.8400e-3 |
| 52 224 | 339 | 2.7059e-3 | 2.7067e-3 | 2.7045e-3 | 2.8612e-3 | 2.8613e-3 | 2.8479e-3 |
| 208 896 | 622 | 2.7053e-3 | 2.7056e-3 | 2.7054e-3 | 2.8599e-3 | 2.8599e-3 | 2.8525e-3 |

On every grid C_f and C_D are within 0.3 % of CFL3D's values on the same grid; on the finest, C_f is
within 0.01 % of CFL3D and FUN3D and C_D equals CFL3D's to 4 digits. With the earlier simplified inflow
(ρ, u fixed, p extrapolated) the results were 0.65 % high: the boundary-layer displacement raised the
inflow pressure by 1.9e-4 p∞, which a total-pressure inflow turns into a 0.34 % lower edge velocity.
The wall-shear order (first cell vs two cells) changes C_f by < 0.1 %.

TMR 2DB bump-in-channel with SA (`sa_bump.py`; same inflow/outflow BCs, symmetry top), C_f from the
two-cell wall derivative, converged to a 1e-10 residual drop:

| N cells | iterations | C_f(0.632) | CFL3D | FUN3D | C_f(0.75) | CFL3D | FUN3D | C_f(0.868) | CFL3D | FUN3D |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 520 | 214 | 5.2647e-3 | 5.3364e-3 | 4.9695e-3 | 5.7111e-3 | 5.7348e-3 | 5.6779e-3 | 2.6805e-3 | 2.7536e-3 | 3.1110e-3 |
| 14 080 | 245 | 5.1701e-3 | 5.2153e-3 | 5.1715e-3 | 6.0455e-3 | 6.0531e-3 | 5.9986e-3 | 2.6595e-3 | 2.7025e-3 | 2.7786e-3 |
| 56 320 | 386 | 5.1600e-3 | 5.1889e-3 | 5.1906e-3 | 6.1254e-3 | 6.1231e-3 | 6.1055e-3 | 2.6601e-3 | 2.6840e-3 | 2.6995e-3 |
| 225 280 | 649 | 5.1618e-3 | 5.1853e-3 | 5.1885e-3 | 6.1479e-3 | 6.1422e-3 | 6.1398e-3 | 2.6628e-3 | 2.6787e-3 | 2.6852e-3 |

TMR finest grid (901 120 cells): 5.185e-3, 6.149e-3, 2.678e-3. C_f(0.75) follows CFL3D on every grid
(within 0.4 %; 0.1 % on 705×321). C_f(0.632) and C_f(0.868) converge to values about 0.45 % and 0.55 %
below the TMR ones (Richardson from the two finest grids: 5.162e-3, 6.155e-3, 2.664e-3). Open: the
cause of these offsets upstream and downstream of the crest; not the limiter (the limited scheme gave
the same fine-grid values) and not the grid. 705×321: 34 min on one CPU core. The 353×161 grid needed
1457 iterations with the former 32-sweep cap (see plan_rans.md, R3, 2DB convergence).

TMR 2DCJ coflowing jet with SA (`sa_jet.py`): M_ref = 0.5, Re = 5e4, two `inflow_total` streams (inner
jet p0/p_ref = 1.1862, T0/T_ref = 1.05; outer 1.046, 1.0), back pressure, zero-thickness plate
(−10 < x < 0, both sides viscous), 4 blocks after patch splitting. C_D = skin friction on both plate
sides / (½ρU²·10); u over a_ref on the centreline (y = 0):

| N cells | iterations | C_D | CFL3D | FUN3D | u(2.716) | CFL3D | FUN3D | u(29.25) | CFL3D | FUN3D | u(95.50) | CFL3D | FUN3D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 280 | 247 | 7.334e-3 | 6.710e-3 | 5.609e-3 | 0.50223 | 0.50220 | 0.50057 | 0.40644 | 0.40750 | 0.39202 | 0.32659 | 0.32680 | 0.32070 |
| 5 120 | 240 | 5.962e-3 | 6.011e-3 | 5.530e-3 | 0.50157 | 0.50165 | 0.50065 | 0.41310 | 0.41385 | 0.40487 | 0.32748 | 0.32765 | 0.32435 |
| 20 480 | 264 | 5.756e-3 | 5.847e-3 | 5.545e-3 | 0.50116 | 0.50121 | 0.50052 | 0.41561 | 0.41602 | 0.41124 | 0.32758 | 0.32768 | 0.32593 |
| 81 920 | 413 | 5.719e-3 | 5.791e-3 | 5.607e-3 | 0.50083 | 0.50081 | 0.50044 | 0.41665 | 0.41685 | 0.41455 | 0.32754 | 0.32759 | 0.32671 |

TMR finest (327 680 cells): C_D 5.766e-3 (CFL3D) / 5.665e-3 (FUN3D); u 0.50060 / 0.50050, 0.41717 /
0.41625, 0.32748 / 0.32712. The centreline velocities follow CFL3D on every grid (within 0.0011 a_ref,
0.0002 on 81 920 cells); u(x, y = 0) is within 0.0008 a_ref of the CFL3D finest-grid profile. C_D
converges to about 5.71e-3, between CFL3D's and FUN3D's finest values (which differ by 1.8 %).
81 920 cells: 482 s on one CPU core.

TMR NACA 0012 (`sa_naca0012.py`; validation-case numerics comparison, SA without point vortex):
M = 0.15, Re_c = 6e6, α = 10°, Riemann far field at ~500 c, family II C-grids (wake cut as a 1-to-1
self-interface). Forces per unit span over ½ρU²c, C_m about c/4 (nose-up positive). CFL3D (SA,
1st-order turbulence advection, as ours) on the same cells; FUN3D/TAU (SA-neg, 2nd order) are listed
at the same node count in the script output:

| grid | cells | iterations | C_L | CFL3D | C_D | CFL3D | C_Dp | CFL3D | C_Dv | CFL3D | C_m | CFL3D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| II.7 | 3 584 | 367 | 1.05426 | – | 0.02441 | – | 0.01903 | – | 0.00538 | – | 0.00282 | – |
| II.6 | 14 336 | 355 | 1.08261 | 1.08291 | 0.014723 | 0.014992 | 0.008726 | 0.008877 | 0.005997 | 0.006116 | 0.00638 | 0.00619 |
| II.5 | 57 344 | 420 | 1.08715 | 1.08735 | 0.012719 | 0.012828 | 0.006596 | 0.006646 | 0.006124 | 0.006183 | 0.00713 | 0.00705 |
| II.4 | 229 376 | 803 | 1.08848 | 1.08850 | 0.012315 | 0.012362 | 0.006155 | 0.006172 | 0.006160 | 0.006190 | 0.00721 | 0.00719 |

TMR infinite-grid bounds (2nd-order turbulence advection): C_L 1.0909–1.0911, C_D 0.012270–0.012275,
C_Dp 0.006060–0.006070, C_Dv 0.006205–0.006206, C_m 0.00676–0.00680. On II.4 C_L agrees with CFL3D on
the same grid to 0.003 %, C_D to 0.4 % (pressure 0.3 %, viscous 0.5 % below), C_m to 0.2 %; the viscous
drag is consistently 0.5–2 % below CFL3D's. II.4: 39 min on one CPU core.
