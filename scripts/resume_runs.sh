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

# CPU cap: pin each run to a single core so it can never use more than 100% CPU. Only the main
# thread is busy (~99% of a core); the OpenMP pool does ~2% of the work, so pinning costs a few
# percent at most. The core is the least busy one on the GPU's NUMA node that does not already
# hold one of our runs. RDBDIFF_PIN_CORE=0 turns pinning off.
PIN_CORE="${RDBDIFF_PIN_CORE:-1}"

# Thread cap per process. Training is not CPU-bound (~80% at 8 threads); 4 keeps an unpinned
# process under the 200% budget, and a pinned one gains nothing from more than 1.
if [ "$PIN_CORE" = 1 ]; then THREADS="${RDBDIFF_THREADS:-1}"; else THREADS="${RDBDIFF_THREADS:-4}"; fi

# Least busy free core on GPU $1's NUMA node, sampled over one second; -1 if none is found.
pick_core() {
  local bus node cpus taken
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1" | cut -d: -f2- | tr 'A-F' 'a-f')
  node=$(cat "/sys/bus/pci/devices/0000:${bus}/numa_node" 2>/dev/null || echo -1)
  if [ "$node" -ge 0 ] 2>/dev/null; then
    cpus=$(cat "/sys/devices/system/node/node${node}/cpulist")
  else
    cpus="0-$(( $(nproc --all) - 1 ))"
  fi
  # a pinned run reports a single core; unpinned processes report a range and are ignored
  taken=$(for p in $(pgrep -u "$USER" -f "python -u main.py"); do
    taskset -cp "$p" 2>/dev/null | awk '{print $NF}'
  done | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
  { awk '/^cpu[0-9]+ /{t=0; for (i = 2; i <= NF; i++) t += $i; print substr($1, 4), $5 + $6, t}' /proc/stat
    sleep 1
    awk '/^cpu[0-9]+ /{t=0; for (i = 2; i <= NF; i++) t += $i; print substr($1, 4), $5 + $6, t}' /proc/stat
  } | awk -v list="$cpus" -v taken=" $taken " '
    BEGIN {
      n = split(list, parts, ",")
      for (i = 1; i <= n; i++) {
        split(parts[i], ab, "-"); hi = (ab[2] == "" ? ab[1] : ab[2])
        for (c = ab[1] + 0; c <= hi + 0; c++) if (index(taken, " " c " ") == 0) ok[c] = 1
      }
      best = -1; core = -1
    }
    ($1 in idle0) {
      d = $3 - tot0[$1]
      if (($1 in ok) && d > 0 && ($2 - idle0[$1]) / d > best) { best = ($2 - idle0[$1]) / d; core = $1 }
      next
    }
    { idle0[$1] = $2; tot0[$1] = $3 }
    END { print core }'
}

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

  PIN=()
  where="GPU ${gpu}"
  if [ "$PIN_CORE" = 1 ]; then
    core=$(pick_core "${gpu}")
    if [ "${core}" -ge 0 ] 2>/dev/null; then
      PIN=(taskset -c "${core}")
      where="${where}, CPU core ${core}"
    fi
  fi

  if [ -f "${workspace}/model_checkpoint/checkpoint.pt" ]; then
    echo "  seed ${seed}: resuming from checkpoint on ${where}"
  else
    echo "  seed ${seed}: starting fresh on ${where}"
  fi

  setsid nohup "${PIN[@]}" env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}" \
    conda run --no-capture-output -n tabsyn python -u main.py \
    "${CONFIG}" \
    --exp_name paper_repro --seed "${seed}" --no_wandb --max_hours "${MAX_HOURS}" \
    "${EXTRA_ARGS[@]}" \
    < /dev/null >> "${LOG_DIR}/paper_repro_${DATASET}_seed${seed}.log" 2>&1 &
  disown
done
