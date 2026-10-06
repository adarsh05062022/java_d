#!/usr/bin/env bash
# Re-score every downstream result that depends on a Berka task, after a change to the task code
# (downstream/tasks.py) or to the feature-level analyses. No GPU is used: it re-reads the
# synthetic databases already in rdbdiff_workspace/.
#
#   setsid nohup scripts/rescore_downstream.sh < /dev/null >> logs/scoring/rescore.log 2>&1 & disown
#
#   TASKS="berka_loan_default berka_card_ownership" scripts/rescore_downstream.sh   # both tasks
#
# TASKS (default: loan default only) selects the downstream scores, cross-domain matrices and the
# DeepSets probe to recompute. The feature-level analyses always cover both tasks. Regimes and
# generator seeds per variant match the result files that already exist, so each file is
# regenerated in place with the same shape. Each job logs to logs/scoring/rescore_<job>.log.

set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root (this script lives in scripts/)

read -r -a TASKS <<< "${TASKS:-berka_loan_default}"
JOBS="${RESCORE_JOBS:-6}"
LOGS=logs/scoring
mkdir -p "$LOGS"

# XGBoost is the CPU-bound stage; 2 threads keeps each process inside the per-process budget.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export RDBDIFF_XGB_JOBS=2 PYTHONUNBUFFERED=1
PY=(conda run --no-capture-output -n tabsyn python -u -m)
FIVE=(real synth mixed_25 mixed_50 mixed)

run() {  # run <job name> <module> [args...]
  local name=$1; shift
  echo "[$(date '+%F %T')] start $name"
  if "${PY[@]}" "$@" > "$LOGS/rescore_${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOGS/rescore_${name}.log)"
  fi
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }

echo "[$(date '+%F %T')] rescore: tasks ${TASKS[*]}, up to $JOBS jobs at a time"

# --- stage 1: independent jobs --------------------------------------------------------------
for ds in berka berka_split berka_split_s1 berka_split_s2 berka_split_s3 berka_split_s5 \
          berka_split_s7 berka_split_reparam berka_split_reparam_s10; do
  throttle; run "$ds" downstream.run --dataset "$ds" --tasks "${TASKS[@]}" --regimes "${FIVE[@]}" &
done
for ds in berka_split_s4 berka_split_s6 berka_split_s12; do   # one generator seed each
  throttle; run "$ds" downstream.run --dataset "$ds" --tasks "${TASKS[@]}" \
    --regimes "${FIVE[@]}" --grdm-seeds 0 &
done
for ds in berka_split_agg berka_split_discrete; do            # every regime
  throttle; run "$ds" downstream.run --dataset "$ds" --tasks "${TASKS[@]}" &
done
for ds in berka_split_repair berka_split_repair_finalonly; do
  throttle; run "$ds" downstream.run --dataset "$ds" --tasks "${TASKS[@]}" --regimes real synth &
done
# substitution ladder on the baseline (read by downstream.ratio_report)
throttle; run berka_split_blend downstream.run --dataset berka_split --tasks "${TASKS[@]}" \
  --regimes real real_50 blend_50 real_25 blend_25 --out-suffix _blend &
throttle; run berka_split_blend50 downstream.run --dataset berka_split --tasks "${TASKS[@]}" \
  --regimes real real_50 blend_50 --out-suffix _blend50 &
throttle; run berka_split_blend75 downstream.run --dataset berka_split --tasks "${TASKS[@]}" \
  --regimes real real_75 blend_75 synth --out-suffix _blend75 &
for ds in berka_split berka_split_agg; do
  throttle; run "crossdomain_$ds" downstream.crossdomain --dataset "$ds" --tasks "${TASKS[@]}" &
done
throttle; run relmlp downstream.relmlp.run --dataset berka_split --tasks "${TASKS[@]}" &
throttle; run feature_compare downstream.feature_compare --dataset berka_split &
throttle; run aggregate_validation downstream.aggregate_validation --dataset berka_split &
throttle; run diagnose_aggregates downstream.diagnose_aggregates &
throttle; run diagnose downstream.diagnose &
wait

# --- stage 2: summaries that read stage-1 output ---------------------------------------------
run feature_provenance downstream.feature_provenance
run ratio_report downstream.ratio_report
run crossdomain_report downstream.crossdomain_report
run report downstream.report

echo "[$(date '+%F %T')] rescore finished"
