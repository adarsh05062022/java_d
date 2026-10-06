#!/usr/bin/env bash
# Relaunch any paper_repro seed that is not currently running, for a given dataset.
# Training resumes from the last epoch-boundary checkpoint; finished seeds are skipped.
# Safe to re-run at any time (e.g. after the ~24h process-lifetime kill on this node).
#
#   scripts/resume_runs.sh                              # resume berka seeds 0 1 2
#   scripts/resume_runs.sh 0 2                          # resume only berka seeds 0 and 2
#   RDBDIFF_DATASET=ccs scripts/resume_runs.sh          # resume ccs seeds 0 1 2
#   RDBDIFF_GPUS="4 6" scripts/resume_runs.sh 1 2       # pin specific GPUs, in order
#
#   # fix experiments (§9): extra main.py flags, own workspace via its own config
#   RDBDIFF_DATASET=berka_split_s2 \
#     RDBDIFF_EXTRA_ARGS="--n_hops 2 --num_neighbors -1 16" scripts/resume_runs.sh 0
#
# Expects configs/<dataset>_fastsample.json to exist.

set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root (this script lives in scripts/)

DATASET="${RDBDIFF_DATASET:-berka}"
CONFIG="configs/${DATASET}_fastsample.json"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 1; }

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)

# Extra main.py flags for the fix experiments (--n_hops, --num_neighbors,
# --seed_balance_alpha, --categorical_encoding). Word-split on purpose.
read -r -a EXTRA_ARGS <<< "${RDBDIFF_EXTRA_ARGS:-}"

# Thread cap per process. Training is not CPU-bound (~80% at 8 threads), so 4 keeps every
# process comfortably under the 200% budget while several runs share the node.
THREADS="${RDBDIFF_THREADS:-4}"

# Wall-clock budget per process, in hours. This node SIGTERMs processes at ~24h17m
# (docs/REPRODUCTION.md 7), so a run must stop and checkpoint itself before that rather than be
# killed mid-epoch. main.py exits 42 once the budget is spent, having written a resume
# point; the watchdog then starts a fresh process that picks up where this one stopped.
MAX_HOURS="${RDBDIFF_MAX_HOURS:-20}"

LOG_DIR="${RDBDIFF_LOG_DIR:-$PWD/logs/train}"
mkdir -p "$LOG_DIR"

# GPU choice is driven by *compute* utilization, not free memory: each run needs only
# ~1.3GB of VRAM, so contention for SMs is what actually slows training. A GPU can sit at
# 0% util with 47GB allocated by an idle neighbour and still be the best host for us.
# Utilization is spiky, so average a few samples rather than trusting one instant.
if [ -n "${RDBDIFF_GPUS:-}" ]; then
  read -r -a FREE_GPUS <<< "${RDBDIFF_GPUS}"
else
  mapfile -t FREE_GPUS < <(
    for _ in 1 2 3 4 5; do
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
      sleep 1
    done | awk -F', ' '
      # skip GPUs without enough headroom for our ~1.3GB plus slack
      { util[$1] += $2; n[$1]++; if ($3 > 135000) full[$1] = 1 }
      END { for (g in util) if (!(g in full)) printf "%d %.1f\n", g, util[g] / n[g] }
    ' | sort -k2 -n | cut -d' ' -f1
  )
fi
echo "dataset: ${DATASET}"
echo "GPU preference order (least busy first): ${FREE_GPUS[*]}"

i=0
for seed in "${SEEDS[@]}"; do
  workspace="rdbdiff_workspace/${DATASET}/paper_repro/seed_${seed}"

  # match on the config too, so seeds of different datasets don't shadow each other
  if pgrep -f -- "${CONFIG} --exp_name paper_repro --seed ${seed} " > /dev/null; then
    echo "  seed ${seed}: already running, skipping"
    continue
  fi

  if [ -f "${workspace}/synthetic_data_sampled_structure_size_multiplier_1.0/eval_result.pkl" ]; then
    echo "  seed ${seed}: already finished (eval_result.pkl exists), skipping"
    continue
  fi

  gpu="${FREE_GPUS[$((i % ${#FREE_GPUS[@]}))]}"
  i=$((i + 1))

  if [ -f "${workspace}/model_checkpoint/checkpoint.pt" ]; then
    echo "  seed ${seed}: resuming from checkpoint on GPU ${gpu}"
  else
    echo "  seed ${seed}: starting fresh on GPU ${gpu}"
  fi

  setsid nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}" \
    conda run --no-capture-output -n tabsyn python -u main.py \
    "${CONFIG}" \
    --exp_name paper_repro --seed "${seed}" --no_wandb --max_hours "${MAX_HOURS}" \
    "${EXTRA_ARGS[@]}" \
    < /dev/null >> "${LOG_DIR}/paper_repro_${DATASET}_seed${seed}.log" 2>&1 &
  disown
done
