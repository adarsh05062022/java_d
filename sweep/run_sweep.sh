#!/usr/bin/env bash
# Unattended driver for the Berka-utility sweep (plan: run-all-i-have-soft-lighthouse).
#
# Reads sweep/manifest.tsv, keeps GPUs filled with training runs (via resume_runs.sh, which
# owns setsid/nohup/checkpoint-resume/--max_hours), decodes reparam variants, scores each
# variant as its seeds finish, and regenerates the report + coverage once everything is done.
#
# Fire-and-forget:
#   setsid nohup ./sweep/run_sweep.sh < /dev/null >> sweep/run_sweep.log 2>&1 & disown
# Stop:
#   kill "$(cat sweep/.pid)"
# Watch:
#   cat sweep/status.md
#
# Survives the node's ~24h SIGTERM the same way watchdog.sh does: a flock guarantees one
# instance, and it hands off to a fresh detached copy every SWEEP_SELF_RESTART_HOURS and on
# TERM. It is idempotent and stateless (all progress lives on disk), so a missed hand-off is
# recovered by simply re-running this script.
#
# Knobs (env):
#   SWEEP_INTERVAL              seconds between ticks            (default 300)
#   SWEEP_SELF_RESTART_HOURS    hand-off cadence                 (default 12)
#   SWEEP_GPU_MEM_MAX_MIB       treat a GPU as usable if used <  (default 120000)
#   SWEEP_RUNS_PER_GPU          co-locate up to N runs per GPU   (default 3)
#   SWEEP_MAX_CONCURRENT        global cap on training runs      (default 24)
#   SWEEP_MAX_HOURS             per-run wall budget              (default 20)
#   SWEEP_GPUS                  restrict to these GPU indices    (default: all visible)

set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root
REPO="$PWD"
SWEEP="$REPO/sweep"
SCRIPT="$SWEEP/run_sweep.sh"

LOG="${SWEEP_LOG:-$SWEEP/run_sweep.log}"
LOCK="${SWEEP_LOCK:-$SWEEP/.lock}"
PIDFILE="${SWEEP_PIDFILE:-$SWEEP/.pid}"
STATUS="$SWEEP/status.md"
MANIFEST="$SWEEP/manifest.tsv"
STATE="$SWEEP/state"
mkdir -p "$STATE"

INTERVAL="${SWEEP_INTERVAL:-300}"
SELF_RESTART_HOURS="${SWEEP_SELF_RESTART_HOURS:-12}"
GPU_MEM_MAX="${SWEEP_GPU_MEM_MAX_MIB:-120000}"  # a GPU is "usable" if used memory is below this
RUNS_PER_GPU="${SWEEP_RUNS_PER_GPU:-3}"         # co-locate up to this many sweep runs per GPU
MAX_CONCURRENT="${SWEEP_MAX_CONCURRENT:-24}"    # global cap on simultaneous training runs
MAX_HOURS="${SWEEP_MAX_HOURS:-20}"
START_TS=$(date +%s)
SLEEP_PID=""

CONDA="conda run --no-capture-output -n tabsyn"
export RDBDIFF_XGB_JOBS="${RDBDIFF_XGB_JOBS:-8}"

# ----- single-instance guard -------------------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] another sweep runner holds the lock; exiting"
  exit 0
fi
echo $$ > "$PIDFILE"

# ----- self hand-off (survives the 24h kill) -------------------------------------------
hand_off() {
  echo "[$(date '+%F %T')] handing off to a fresh runner ($1)"
  [ -n "${SLEEP_PID:-}" ] && kill "$SLEEP_PID" 2>/dev/null
  exec 9>&-                                  # drop the lock before the successor starts
  setsid nohup env \
    SWEEP_LOG="$LOG" SWEEP_LOCK="$LOCK" SWEEP_PIDFILE="$PIDFILE" \
    SWEEP_INTERVAL="$INTERVAL" SWEEP_SELF_RESTART_HOURS="$SELF_RESTART_HOURS" \
    SWEEP_GPU_MEM_MAX_MIB="$GPU_MEM_MAX" SWEEP_RUNS_PER_GPU="$RUNS_PER_GPU" \
    SWEEP_MAX_CONCURRENT="$MAX_CONCURRENT" \
    SWEEP_MAX_HOURS="$MAX_HOURS" ${SWEEP_GPUS:+SWEEP_GPUS="$SWEEP_GPUS"} \
    "$SCRIPT" < /dev/null >> "$LOG" 2>&1 &
  disown
  exit 0
}
trap 'hand_off "SIGTERM"' TERM

echo "[$(date '+%F %T')] sweep runner started (pid $$); interval ${INTERVAL}s; gpu_mem_max ${GPU_MEM_MAX}MiB; runs_per_gpu ${RUNS_PER_GPU}; max_concurrent ${MAX_CONCURRENT}"

# ----- helpers -------------------------------------------------------------------------
run_dir() {  # name seed [variant-suffix]
  echo "rdbdiff_workspace/$1/paper_repro/seed_$2/synthetic_data_sampled_structure_size_multiplier_1.0${3:-}"
}
is_sampled() {  # name seed  -> training + sampling + fidelity-eval all finished
  [ -f "$(run_dir "$1" "$2")/eval_result.pkl" ]
}
is_running() {  # name seed
  pgrep -f -- "configs/$1_fastsample.json --exp_name paper_repro --seed $2 " >/dev/null 2>&1
}
my_train_pids() {
  # the python child only -- not the `conda run` wrapper, which also carries this cmdline
  pgrep -af -- "main\.py .*_fastsample.json --exp_name paper_repro --seed " 2>/dev/null \
    | grep -v "conda run" | awk '{print $1}'
}
gpu_of_pid() {
  tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1
}
gpus_in_use_by_sweep() {
  local p g out=""
  for p in $(my_train_pids); do
    g=$(gpu_of_pid "$p"); [ -n "$g" ] && out="$out $g"
  done
  echo "$out" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' '
}
# "<gpu> <count>" for every GPU currently hosting one or more sweep training runs
gpu_run_counts() {
  local p g
  for p in $(my_train_pids); do g=$(gpu_of_pid "$p"); [ -n "$g" ] && echo "$g"; done \
    | grep -E '^[0-9]+$' | sort -n | uniq -c | awk '{print $2, $1}'
}
# emits each usable GPU index once per free co-location slot (RUNS_PER_GPU minus current load)
candidate_gpus() {
  local allow="${SWEEP_GPUS:-}" idx mem cnt slots k
  declare -A used
  while read -r gi gc; do [ -n "$gi" ] && used[$gi]=$gc; done < <(gpu_run_counts)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | while IFS=',' read -r idx mem; do
      idx=$(echo "$idx" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
      [ -n "$allow" ] && ! grep -qw "$idx" <<< "$allow" && continue
      [ "${mem:-999999}" -lt "$GPU_MEM_MAX" ] || continue
      cnt=${used[$idx]:-0}
      slots=$(( RUNS_PER_GPU - cnt ))
      [ "$slots" -gt 0 ] || continue
      for ((k = 0; k < slots; k++)); do echo "$idx"; done
    done
}
running_count() { my_train_pids | wc -l | tr -d ' '; }

# manifest -> arrays (skip # and blank lines)
NAMES=(); SEEDS=(); KINDS=(); FLAGS=()
load_manifest() {
  NAMES=(); SEEDS=(); KINDS=(); FLAGS=()
  local name seeds kind flags
  while IFS=$'\t' read -r name seeds kind flags; do
    [ -z "${name:-}" ] && continue
    case "$name" in \#*) continue ;; esac
    NAMES+=("$name"); SEEDS+=("$seeds"); KINDS+=("$kind"); FLAGS+=("${flags:-}")
  done < "$MANIFEST"
}

variant_suffix() {  # kind -> sample dir suffix used by scoring
  [ "$1" = "berka_reparam" ] && echo "_decoded" || echo ""
}

all_seeds_sampled() {  # name "s1,s2,.."
  local n="$1" s
  for s in ${2//,/ }; do is_sampled "$n" "$s" || return 1; done
  return 0
}

score_variant() {  # name "seeds" kind flags
  local name="$1" seedcsv="$2" kind="$3"
  local seeds="${seedcsv//,/ }" suf; suf=$(variant_suffix "$kind")
  echo "[$(date '+%F %T')] scoring $name (seeds $seeds, kind $kind)"

  if [ "$kind" = "berka_reparam" ]; then
    $CONDA python -m dep_fidelity.decode_run --dataset "$name" --seeds $seeds \
      >> "$REPO/logs/scoring/decode_${name}.log" 2>&1 9>&- || {
        echo "[$(date '+%F %T')] decode_run failed for $name; will retry next tick"; return 1; }
  fi

  $CONDA python -m downstream.run --dataset "$name" --grdm-seeds $seeds \
    >> "$REPO/logs/scoring/downstream_${name}.log" 2>&1 9>&- || {
      echo "[$(date '+%F %T')] downstream.run failed for $name; will retry next tick"; return 1; }

  case "$name" in
    berka*)
      local args=()
      for s in $seeds; do args+=(--synthetic "$(run_dir "$name" "$s" "$suf")/synthetic_database"); done
      $CONDA python -m dep_fidelity.run "${args[@]}" --label "$name" \
        >> "$REPO/logs/scoring/depfid_${name}.log" 2>&1 9>&- || \
        echo "[$(date '+%F %T')] dep_fidelity.run failed for $name (non-fatal)"
      ;;
  esac

  touch "$STATE/${name}.scored"
  echo "[$(date '+%F %T')] $name scored"
}

write_status() {
  local now tick_gpus n s
  now=$(date '+%F %T'); tick_gpus=$(gpus_in_use_by_sweep)
  {
    echo "# Sweep status — $now"
    echo
    echo "runner pid $(cat "$PIDFILE" 2>/dev/null), $(running_count) training run(s) live, sweep GPUs: ${tick_gpus:-none}"
    echo
    echo "| variant | kind | seeds trained | scored | task A | task B |"
    echo "|---|---|---|---|---|---|"
    for i in "${!NAMES[@]}"; do
      n="${NAMES[$i]}"
      local trained="" done_n=0 tot=0
      for s in ${SEEDS[$i]//,/ }; do
        tot=$((tot+1))
        if is_sampled "$n" "$s"; then done_n=$((done_n+1))
        elif is_running "$n" "$s"; then
          local st; st=$(grep -oE "Step: [0-9]+/[0-9]+" "$REPO/logs/train/paper_repro_${n}_seed${s}.log" 2>/dev/null | tail -1)
          local sm; sm=$(grep -oE "Sampling timestep [0-9]+/[0-9]+" "$REPO/logs/train/paper_repro_${n}_seed${s}.log" 2>/dev/null | tail -1)
          trained="$trained s${s}:${sm:-${st:-run}}"
        fi
      done
      local scored="no"; [ -f "$STATE/${n}.scored" ] && scored="yes"
      local reta="-" retb="-"
      if [ -f "$STATE/${n}.scored" ]; then
        read -r _ reta retb _ < <(python3 "$SWEEP/summarize.py" "$n" 2>/dev/null)
      fi
      echo "| $n | ${KINDS[$i]} | ${done_n}/${tot}${trained:+ (${trained# })} | $scored | ${reta:--} | ${retb:--} |"
    done
    echo
    [ -f "$SWEEP/.complete" ] && cat "$SWEEP/.complete"
    true
  } > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
}

# ----- main loop --------------------------------------------------------------------------
while true; do
  load_manifest

  # 1. schedule training onto free GPUs
  mapfile -t CANDS < <(candidate_gpus)
  gi=0
  rc=$(running_count)
  for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"; fl="${FLAGS[$i]}"
    for s in ${SEEDS[$i]//,/ }; do
      is_sampled "$n" "$s" && continue
      is_running "$n" "$s" && continue
      [ "$rc" -ge "$MAX_CONCURRENT" ] && break 2
      [ "$gi" -lt "${#CANDS[@]}" ] || break 2
      g="${CANDS[$gi]}"; gi=$((gi+1)); rc=$((rc+1))
      echo "[$(date '+%F %T')] launch $n seed $s on GPU $g"
      # 9>&- is load-bearing: without it resume_runs.sh's detached training jobs inherit the
      # lock fd and hold the flock for their whole ~20h lifetime (see watchdog.sh).
      RDBDIFF_DATASET="$n" RDBDIFF_GPUS="$g" RDBDIFF_EXTRA_ARGS="$fl" \
        RDBDIFF_MAX_HOURS="$MAX_HOURS" scripts/resume_runs.sh "$s" \
        >> "$REPO/logs/infra/sweep_launch.log" 2>&1 9>&-
    done
  done

  # 2. score any variant whose seeds are all sampled and not yet scored
  for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"
    [ -f "$STATE/${n}.scored" ] && continue
    all_seeds_sampled "$n" "${SEEDS[$i]}" || continue
    score_variant "$n" "${SEEDS[$i]}" "${KINDS[$i]}" "${FLAGS[$i]}"
  done

  # 3. status page
  write_status

  # 4. done?
  all_scored=1
  for i in "${!NAMES[@]}"; do [ -f "$STATE/${NAMES[$i]}.scored" ] || all_scored=0; done
  if [ "$all_scored" -eq 1 ]; then
    echo "[$(date '+%F %T')] every manifest variant scored; finalising"
    $CONDA python -m downstream.report >> "$REPO/logs/scoring/report_regen.log" 2>&1 9>&-
    $CONDA python -m downstream.coverage > "$SWEEP/coverage.txt" 2>&1 9>&-
    {
      echo "## SWEEP COMPLETE — $(date '+%F %T')"
      echo
      echo "RESULTS_downstream.md regenerated; coverage in sweep/coverage.txt."
    } > "$SWEEP/.complete"
    write_status
    rm -f "$PIDFILE"
    echo "[$(date '+%F %T')] sweep runner exiting"
    exit 0
  fi

  # 5. retire before the node's ~24h SIGTERM can reach us
  if [ $(( $(date +%s) - START_TS )) -ge $(( SELF_RESTART_HOURS * 3600 )) ]; then
    hand_off "reached ${SELF_RESTART_HOURS}h"
  fi

  sleep "$INTERVAL" 9>&- &
  SLEEP_PID=$!
  wait "$SLEEP_PID" || true
done
