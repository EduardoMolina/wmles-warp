#!/usr/bin/env bash
# WMLES channel Re_tau = 5200 sweep on a single GPU (e.g. g5.4xlarge, 1x A10G 24 GB).
#
# The channel grids are small (98k-196k cells), so one case leaves the GPU mostly idle (the run is
# bound by kernel-launch/Python overhead). The script therefore runs up to MAX_JOBS cases
# concurrently on the same GPU and starts the next queued case when one finishes. Cases are run in
# the order listed (most important first). A case whose run.log already says "finished" is skipped,
# so the script can simply be re-run after an interruption.
#
#   nohup bash verification/channel_sweep_1gpu.sh > runs/sweep.log 2>&1 &   # queue in background
#   MAX_JOBS=2 STEPS=100000 bash verification/channel_sweep_1gpu.sh
#   ONLY="ar1_eq_c3 ar1_alg_c3" bash verification/channel_sweep_1gpu.sh   # subset of cases
#   DRY=1 bash verification/channel_sweep_1gpu.sh                          # generate + `info` only
#
# Each case: runs/channel_<name>/{case.yaml, run.log, out/}. Run from the repository root with the
# `pyvista` env active.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${GPU:-0}"
MAX_JOBS="${MAX_JOBS:-3}"
PRECISION="${PRECISION:-float32}"
STEPS="${STEPS:-200000}"
STATS_START="${STATS_START:-50000}"
ONLY="${ONLY:-}"
DRY="${DRY:-0}"

#      name        grid                         model           cell_index
CASES=(
  "ar1_eq_c3   channel_ar1_1dwm_129x33.p3d equilibrium_ode 3"
  "ar1_alg_c3  channel_ar1_1dwm_129x33.p3d algebraic       3"
  "ar2_eq_c3   channel_ar2_1dwm_65x33.p3d  equilibrium_ode 3"
  "ar1_eq_c2   channel_ar1_1dwm_129x33.p3d equilibrium_ode 2"
  "ar1_eq_c4   channel_ar1_1dwm_129x33.p3d equilibrium_ode 4"
  "ar1_eq_c1   channel_ar1_1dwm_129x33.p3d equilibrium_ode 1"
  "ar1_eq_c5   channel_ar1_1dwm_129x33.p3d equilibrium_ode 5"
)

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$ROOT/runs"

running() { jobs -rp | wc -l | tr -d ' '; }

for c in "${CASES[@]}"; do
  read -r name grid model cell <<<"$c"
  if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then continue; fi
  dir="$ROOT/runs/channel_$name"
  if [[ -f "$dir/run.log" ]] && grep -q "^finished" "$dir/run.log"; then
    echo "$(date '+%F %T') skip $name (already finished)"; continue
  fi
  mkdir -p "$dir"
  sed -e "s|file: channel_ar1_1dwm_129x33.p3d|file: $ROOT/cases/channel/$grid|" \
      -e "s|precision: float64|precision: $PRECISION|" \
      -e "s|type: equilibrium_ode|type: $model|" \
      -e "s|exchange: {cell_index: 3}|exchange: {cell_index: $cell}|" \
      -e "s|^run: .*|run: {steps: $STEPS, output_every: 50000, stats_start: $STATS_START, stats_every: 5, restart_every: 50000, log_every: 2000}|" \
      "$ROOT/cases/channel/case.yaml" > "$dir/case.yaml"
  if [[ "$DRY" == "1" ]]; then
    echo "== $name"; (cd "$dir" && python -m wmles_wrap info case.yaml | grep -E "precision|parent blocks|devices")
    continue
  fi
  while (( $(running) >= MAX_JOBS )); do sleep 30; done
  (cd "$dir" && python -u -m wmles_wrap run case.yaml > run.log 2>&1) &
  echo "$(date '+%F %T') started $name on GPU $GPU -> $dir/run.log"
  sleep 5   # stagger startup (kernel compilation / grid setup)
done

wait
echo "$(date '+%F %T') all cases done"
grep -H "^finished\|Error\|Traceback" "$ROOT"/runs/channel_*/run.log || true
