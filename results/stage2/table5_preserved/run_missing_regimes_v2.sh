#!/usr/bin/env bash
# Re-run of the 7 missing-regime California downstream evals (real_25/50/75, blend_25/75,
# mixed_25/50) for the 3 aggregate-conditioning-fine-tune arms. The first attempt (launched
# 2026-10-13 15:42) died silently ~37 min in with no error/traceback/OOM evidence and no output
# file -- most likely its process group was tied to a Bash-tool session that got torn down.
# This version fully detaches with setsid + disown + stdin from /dev/null so it survives
# regardless of this tool-call's own lifecycle. No training/resampling: reuses already-sampled
# synthetic data and checkpoints under rdbdiff_workspace/. Writes to NEW *_missing.json files via
# --out-suffix so the existing 3-regime (real/synth/mixed/blend_50) results are untouched.
set -uo pipefail
cd /storage/s25017/AMEX/java_d-calagg-ft

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export RDBDIFF_XGB_JOBS=2 PYTHONUNBUFFERED=1

OUTDIR=experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/downstream
LOGDIR=experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/logs
mkdir -p "$LOGDIR"

MISSING_REGIMES=(real_25 real_50 real_75 blend_25 blend_75 mixed_25 mixed_50)

run_arm() {
  local ds=$1
  echo "[$(date '+%F %T')] start $ds (pid $$)"
  conda run --no-capture-output -n tabsyn python -u -m downstream.run \
    --dataset "$ds" --tasks california_inctot california_empstat \
    --regimes "${MISSING_REGIMES[@]}" \
    --out-dir "$OUTDIR" --out-suffix _missing \
    > "$LOGDIR/downstream_${ds}_missing.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] done  $ds (exit $rc)"
}

{
  echo "[$(date '+%F %T')] relaunch (attempt 2) of 3 arms in parallel: california_agg_ft, california_agg_ft_label_free, california_ft_control"
  run_arm california_agg_ft &
  pid1=$!
  run_arm california_agg_ft_label_free &
  pid2=$!
  run_arm california_ft_control &
  pid3=$!
  echo "child pids: $pid1 $pid2 $pid3" > "$LOGDIR/missing_regimes_v2.pids"
  wait
  echo "[$(date '+%F %T')] all arms finished"
} >> "$LOGDIR/missing_regimes_master_v2.log" 2>&1
