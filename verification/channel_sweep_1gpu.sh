#!/usr/bin/env bash
# WMLES channel Re_tau = 5200 sweep on a single GPU (e.g. g5.4xlarge, 1x A10G 24 GB).
#
# Runs up to MAX_JOBS cases concurrently on the same GPU (the small channel grids leave a GPU mostly
# idle) and starts the next queued case when one finishes. Existing runs are never overwritten:
#   - a case whose run.log says "finished" is skipped;
#   - a case whose run.log exists but did not finish is skipped with a warning (FORCE=1 reruns it).
# So the script can simply be re-run after adding cases or after an interruption.
#
#   mkdir -p runs && nohup bash verification/channel_sweep_1gpu.sh > runs/sweep.log 2>&1 &
#   ONLY="ar1_257x65_eq_c3 ar2_129x65_eq_c3" bash verification/channel_sweep_1gpu.sh
#   MAX_JOBS=2 bash verification/channel_sweep_1gpu.sh
#   DRY=1 bash verification/channel_sweep_1gpu.sh        # generate case.yaml + `info` only
#
# Per grid (from the file name channel_<ar>_1dwm_<ni>x<nj>.p3d, as in cases/channel/extrude.f90):
#   nk    = (ni-1)*4.8/12.8 + 1, doubled for ar2 (dz = dx for ar1, dz = dx/2 for ar2);
#   steps = STEPS*(nj-1)/32 and stats start = STATS_START*(nj-1)/32, because dt scales with the
#           wall-normal spacing; every grid then covers the same physical time (~t = 500 h/U_b).
# Exchange: cN = cell_index N; dX = distance X (first cell centre at or beyond X from the wall).
#   d0.095 selects h = 0.098 / 0.104 / 0.106 (cell 4 / 6 / 8) on the ny = 33 / 49 / 65 grids.
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
FORCE="${FORCE:-0}"

#      name               grid                         model           exchange
CASES=(
  # base grids (first sweep)
  "ar1_eq_c3          channel_ar1_1dwm_129x33.p3d equilibrium_ode c3"
  "ar1_alg_c3         channel_ar1_1dwm_129x33.p3d algebraic       c3"
  "ar2_eq_c3          channel_ar2_1dwm_65x33.p3d  equilibrium_ode c3"
  "ar1_eq_c2          channel_ar1_1dwm_129x33.p3d equilibrium_ode c2"
  "ar1_eq_c4          channel_ar1_1dwm_129x33.p3d equilibrium_ode c4"
  "ar1_eq_c1          channel_ar1_1dwm_129x33.p3d equilibrium_ode c1"
  "ar1_eq_c5          channel_ar1_1dwm_129x33.p3d equilibrium_ode c5"
  # grid refinement: fixed cell index 3 and fixed distance (~0.1 h); largest first
  "ar1_257x65_eq_c3   channel_ar1_1dwm_257x65.p3d equilibrium_ode c3"
  "ar1_257x65_eq_d095 channel_ar1_1dwm_257x65.p3d equilibrium_ode d0.095"
  "ar2_129x65_eq_c3   channel_ar2_1dwm_129x65.p3d equilibrium_ode c3"
  "ar2_129x65_eq_d095 channel_ar2_1dwm_129x65.p3d equilibrium_ode d0.095"
  "ar1_193x49_eq_c3   channel_ar1_1dwm_193x49.p3d equilibrium_ode c3"
  "ar1_193x49_eq_d095 channel_ar1_1dwm_193x49.p3d equilibrium_ode d0.095"
  "ar2_97x49_eq_c3    channel_ar2_1dwm_97x49.p3d  equilibrium_ode c3"
  "ar2_97x49_eq_d095  channel_ar2_1dwm_97x49.p3d  equilibrium_ode d0.095"
  "ar2_65x33_eq_d095  channel_ar2_1dwm_65x33.p3d  equilibrium_ode d0.095"
)

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$ROOT/runs"

running() { jobs -rp | wc -l | tr -d ' '; }
stamp() { date '+%F %T'; }

for c in "${CASES[@]}"; do
  read -r name grid model exch <<<"$c"
  if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then continue; fi
  dir="$ROOT/runs/channel_$name"
  if [[ -f "$dir/run.log" && "$DRY" != "1" ]]; then
    if grep -q "^finished" "$dir/run.log"; then echo "$(stamp) skip $name (finished)"; continue; fi
    if [[ "$FORCE" != "1" ]]; then echo "$(stamp) skip $name (run.log exists, not finished; FORCE=1 reruns)"; continue; fi
  fi
  [[ -f "$ROOT/cases/channel/$grid" ]] || { echo "$(stamp) missing grid $grid, skip $name"; continue; }
  [[ "$grid" =~ _([0-9]+)x([0-9]+)\.p3d$ ]] || { echo "bad grid name $grid"; exit 1; }
  ni=${BASH_REMATCH[1]}; nj=${BASH_REMATCH[2]}
  nk=$(( (ni - 1) * 3 / 8 + 1 ))
  [[ "$grid" == *_ar2_* ]] && nk=$(( 2 * (nk - 1) + 1 ))
  steps=$(( STEPS * (nj - 1) / 32 )); stats=$(( STATS_START * (nj - 1) / 32 ))
  case "$exch" in
    c*) exyaml="cell_index: ${exch#c}" ;;
    d*) exyaml="distance: ${exch#d}" ;;
    *) echo "bad exchange $exch"; exit 1 ;;
  esac
  mkdir -p "$dir"
  sed -e "s|file: channel_ar1_1dwm_129x33.p3d|file: $ROOT/cases/channel/$grid|" \
      -e "s|extrude: {nk: 49, length: 4.8}|extrude: {nk: $nk, length: 4.8}|" \
      -e "s|precision: float64|precision: $PRECISION|" \
      -e "s|type: equilibrium_ode|type: $model|" \
      -e "s|exchange: {cell_index: 3}|exchange: {$exyaml}|" \
      -e "s|^run: .*|run: {steps: $steps, output_every: 50000, stats_start: $stats, stats_every: 5, restart_every: 50000, log_every: 2000}|" \
      "$ROOT/cases/channel/case.yaml" > "$dir/case.yaml"
  cells=$(( (ni - 1) * (nj - 1) * (nk - 1) ))
  if [[ "$DRY" == "1" ]]; then
    echo "== $name  nk $nk  cells $cells  steps $steps  stats_start $stats  exchange {$exyaml}"
    (cd "$dir" && python -m wmles_wrap info case.yaml | grep -E "parent blocks|total cells")
    continue
  fi
  while (( $(running) >= MAX_JOBS )); do sleep 30; done
  (cd "$dir" && python -u -m wmles_wrap run case.yaml > run.log 2>&1) &
  echo "$(stamp) started $name ($cells cells, $steps steps, exchange {$exyaml}) on GPU $GPU"
  sleep 5   # stagger startup (kernel compilation / grid setup)
done

wait
echo "$(stamp) all cases done"
grep -H "^finished\|Error\|Traceback" "$ROOT"/runs/channel_*/run.log || true
