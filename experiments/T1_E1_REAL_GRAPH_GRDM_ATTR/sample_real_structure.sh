#!/usr/bin/env bash
# T1_E1: generate GRDM attributes on the REAL training graph (no retraining).
#
# For each (dataset, seed) this
#   1. creates an isolated workspace  rdbdiff_workspace/<dataset>/T1_E1_real_structure/seed_<N>/
#      holding byte-identical copies of the paper_repro run's trained weights and normalizers,
#      so nothing under paper_repro/ is written (main.py rewrites <workspace>/real_database);
#   2. launches upstream main.py with --fixed_structure, i.e. GRDM's own ablation path: the
#      real graph is kept, every non-dimension node's x is zeroed, replaced by N(0, I) noise and
#      denoised by the trained EMA model. Output lands in
#      <workspace>/synthetic_data_real_structure/synthetic_database/.
#
# All other main.py arguments are the ones the paper_repro baseline was launched with
# (n_hops 1, num_neighbors -1, default ordinal/gaussian categoricals, no fix flags).
#
#   experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/sample_real_structure.sh berka_split 0 1 2
#   RDBDIFF_GPUS="6 3 4" experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/sample_real_structure.sh california 0 1 2

set -uo pipefail
cd "$(dirname "$0")/../.."                    # repo root

DATASET="$1"; shift
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
CONFIG="configs/${DATASET}_fastsample.json"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 1; }

EXP_NAME="T1_E1_real_structure"
THREADS="${RDBDIFF_THREADS:-4}"
MAX_HOURS="${RDBDIFF_MAX_HOURS:-20}"
LOG_DIR="$PWD/results/T1_E1_REAL_GRAPH_GRDM_ATTR/logs"
mkdir -p "$LOG_DIR"

if [ -n "${RDBDIFF_GPUS:-}" ]; then
  read -r -a GPUS <<< "${RDBDIFF_GPUS}"
else
  # every GPU on this node is near 100% util, so rank by free memory instead
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | cut -d, -f1)
fi

i=0
for seed in "${SEEDS[@]}"; do
  src="rdbdiff_workspace/${DATASET}/paper_repro/seed_${seed}"
  dst="rdbdiff_workspace/${DATASET}/${EXP_NAME}/seed_${seed}"
  [ -f "${src}/model_checkpoint/model_ema.pt" ] || { echo "no trained model in ${src}" >&2; exit 1; }

  if [ -d "${dst}/synthetic_data_real_structure/synthetic_database" ]; then
    echo "  ${DATASET} seed ${seed}: already sampled, skipping"
    continue
  fi
  if pgrep -f -- "${CONFIG} --exp_name ${EXP_NAME} --seed ${seed} " > /dev/null; then
    echo "  ${DATASET} seed ${seed}: already running, skipping"
    continue
  fi

  mkdir -p "${dst}/model_checkpoint" "${dst}/normalizers"
  # model.pt + model_ema.pt are all main.py reads once model_ema.pt exists (training is skipped);
  # checkpoint.pt (optimizer state) is deliberately not copied so a resume path can never trigger
  for f in model.pt model_ema.pt loss.csv; do
    [ -f "${dst}/model_checkpoint/${f}" ] || cp -p "${src}/model_checkpoint/${f}" "${dst}/model_checkpoint/${f}"
  done
  for f in "${src}"/normalizers/*.pkl; do
    [ -f "${dst}/normalizers/$(basename "$f")" ] || cp -p "$f" "${dst}/normalizers/"
  done

  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  i=$((i + 1))
  echo "  ${DATASET} seed ${seed}: sampling on GPU ${gpu} -> ${dst}"

  setsid nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}" \
    conda run --no-capture-output -n tabsyn python -u main.py \
    "${CONFIG}" \
    --exp_name "${EXP_NAME}" --seed "${seed}" --fixed_structure --no_wandb \
    --max_hours "${MAX_HOURS}" \
    < /dev/null >> "${LOG_DIR}/sample_${DATASET}_seed${seed}.log" 2>&1 &
  disown
done
