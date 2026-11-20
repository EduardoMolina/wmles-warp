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
