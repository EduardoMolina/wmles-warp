#!/usr/bin/env bash
# WMLES channel Re_tau = 5200 sweep: one case per GPU, run concurrently (the channel grids are far
# too small to benefit from splitting one case over several GPUs).
#
#   bash verification/channel_sweep.sh            # generate runs/<name>/case.yaml and launch
#   DRY=1 bash verification/channel_sweep.sh      # generate and run `info` only (works on CPU)
#   STEPS=200000 PRECISION=float32 bash verification/channel_sweep.sh
#
# Each case: runs/<name>/{case.yaml, run.log, out/}. Run from the repository root with the
# `pyvista` env active. Edit CASES below to change the sweep (gpu name grid model cell_index).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRECISION="${PRECISION:-float32}"
STEPS="${STEPS:-200000}"
STATS_START="${STATS_START:-50000}"
DRY="${DRY:-0}"

CASES=(
  "0 ar1_eq_c3   channel_ar1_1dwm_129x33.p3d equilibrium_ode 3"
  "1 ar1_alg_c3  channel_ar1_1dwm_129x33.p3d algebraic       3"
  "2 ar2_eq_c3   channel_ar2_1dwm_65x33.p3d  equilibrium_ode 3"
  "3 ar1_eq_c1   channel_ar1_1dwm_129x33.p3d equilibrium_ode 1"
  "4 ar1_eq_c2   channel_ar1_1dwm_129x33.p3d equilibrium_ode 2"
  "5 ar1_eq_c4   channel_ar1_1dwm_129x33.p3d equilibrium_ode 4"
  "6 ar1_eq_c5   channel_ar1_1dwm_129x33.p3d equilibrium_ode 5"
)

export PYTHONPATH="$ROOT/src"
for c in "${CASES[@]}"; do
  read -r gpu name grid model cell <<<"$c"
  dir="$ROOT/runs/channel_$name"
  mkdir -p "$dir"
  sed -e "s|file: channel_ar1_1dwm_129x33.p3d|file: $ROOT/cases/channel/$grid|" \
      -e "s|precision: float64|precision: $PRECISION|" \
      -e "s|type: equilibrium_ode|type: $model|" \
      -e "s|exchange: {cell_index: 3}|exchange: {cell_index: $cell}|" \
      -e "s|^run: .*|run: {steps: $STEPS, output_every: 50000, stats_start: $STATS_START, stats_every: 5, restart_every: 50000, log_every: 2000}|" \
      "$ROOT/cases/channel/case.yaml" > "$dir/case.yaml"
  if [[ "$DRY" == "1" ]]; then
    echo "== $name (gpu $gpu)"; (cd "$dir" && python -m wmles_wrap info case.yaml | grep -E "precision|parent blocks|devices")
  else
    (cd "$dir" && CUDA_VISIBLE_DEVICES=$gpu nohup python -u -m wmles_wrap run case.yaml > run.log 2>&1 &)
    echo "launched $name on GPU $gpu -> $dir/run.log"
  fi
done
