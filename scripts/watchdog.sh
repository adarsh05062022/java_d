#!/usr/bin/env bash
# Keep the fix experiments alive across this node's ~24h process kill (docs/REPRODUCTION.md 7).
#
# Two layers, because the node SIGTERMs *every* long-lived process at ~24h17m:
#   1. Each run gets --max_hours (default 20 via resume_runs.sh), so it checkpoints and exits
#      with code 42 before the SIGTERM can land mid-epoch. Nothing is ever lost to a kill.
#   2. This watchdog relaunches anything not running and not finished. resume_runs.sh is
#      idempotent, so re-running it on a timer is enough.
#
# The watchdog is itself a long-lived process, so it would be SIGTERM'd at ~24h too, and then
# nothing would relaunch the runs. It therefore hands off to a fresh detached copy of itself
# every SELF_RESTART_HOURS (resetting its process age), and also traps SIGTERM to hand off
# immediately if the node gets to it first. A flock guarantees only one instance is ever live.
#
#   setsid nohup scripts/watchdog.sh < /dev/null >> logs/infra/watchdog.log 2>&1 & disown
#   pkill -f watchdog.sh    # stop it
#
# Each entry pins its own GPU. Without pinning, resume_runs.sh ranks GPUs independently per
# launch and a just-started job has not registered any utilization yet, so several launches in a
# row all pick the same "least busy" device -- which is exactly how S2/S3/S4 ended up sharing
# GPU 5 and running at half speed.
#
# Override with RDBDIFF_WATCH="dataset:gpu:seeds:extra args|...".

set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root (this script lives in scripts/)

SCRIPT="$PWD/scripts/watchdog.sh"
LOG="${RDBDIFF_WATCH_LOG:-$PWD/logs/infra/watchdog.log}"
LOCK="${RDBDIFF_WATCH_LOCK:-$PWD/logs/infra/.watchdog.lock}"
mkdir -p "$(dirname "$LOG")"

# Single-instance guard. Held for this process's lifetime; released automatically on exit,
# including on SIGKILL, so a dead watchdog never blocks its successor.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] another watchdog already holds the lock; exiting"
  exit 0
fi

# Record the live pid so it can be found without pgrep-matching (a pgrep pattern containing
# "watchdog.sh" also matches whatever shell is doing the searching).
PIDFILE="${RDBDIFF_WATCH_PIDFILE:-$PWD/logs/infra/.watchdog.pid}"
echo $$ > "$PIDFILE"

SLEEP_PID=""
INTERVAL="${RDBDIFF_WATCH_INTERVAL:-600}"
SELF_RESTART_HOURS="${RDBDIFF_WATCH_SELF_RESTART_HOURS:-12}"
START_TS=$(date +%s)

# Per-entry seeds: "dataset:gpu:seeds:extra args". GPUs are chosen by measured contention, one
# job per device -- co-locating two runs on one GPU cost ~2.4x throughput when S2/S3/S4 all
# landed on GPU 5.
#
# S6/S7 push conditioning depth to 3 hops, which docs/downstream/DIAGNOSIS.md A2 predicts is what the card task
# needs (the effective window sits one hop behind the nominal depth). num_neighbors caps the
# fan-out past hop 1: "-1 16 8" keeps hops 1-2 identical to S2/S5, so the third hop is the only
# variable. "-1 64 8" was measured at ~2x the step cost and rejected.
#
# S12 is the first entry that changes the denoiser rather than its inputs: multi-statistic
# aggregation + residual layers + exact per-relation cardinality, layered on S5. california_s7
# finished and has been dropped from the list, freeing GPU 1 for it.
DEFAULT_WATCH='berka_split_s7:3:0:--n_hops 3 --num_neighbors -1 16 8 --seed_balance_alpha 0.5|berka_split_s7:7:1:--n_hops 3 --num_neighbors -1 16 8 --seed_balance_alpha 0.5|berka_split_s7:6:2:--n_hops 3 --num_neighbors -1 16 8 --seed_balance_alpha 0.5|berka_split_s6:5:0:--n_hops 3 --num_neighbors -1 16 8|berka_split_s5:4:1:--n_hops 2 --num_neighbors -1 16 --seed_balance_alpha 0.5|berka_split_s5:2:2:--n_hops 2 --num_neighbors -1 16 --seed_balance_alpha 0.5|berka_split_s12:1:0:--n_hops 2 --num_neighbors -1 16 --seed_balance_alpha 0.5 --gnn_aggr sum mean max std --gnn_residual --degree_features'
WATCH="${RDBDIFF_WATCH:-$DEFAULT_WATCH}"

# Hand off to a fresh detached copy and exit. The lock fd is closed first so the successor can
# take it immediately rather than racing this process's teardown.
hand_off() {
  echo "[$(date '+%F %T')] handing off to a fresh watchdog ($1)"
  # drop the lock completely before spawning: close our fd and stop the sleep child that
  # would otherwise still be holding a copy of it
  [ -n "${SLEEP_PID:-}" ] && kill "$SLEEP_PID" 2>/dev/null
  exec 9>&-
  setsid nohup env RDBDIFF_WATCH="$WATCH" RDBDIFF_WATCH_INTERVAL="$INTERVAL" \
    RDBDIFF_WATCH_SELF_RESTART_HOURS="$SELF_RESTART_HOURS" \
    RDBDIFF_WATCH_PIDFILE="$PIDFILE" RDBDIFF_WATCH_LOCK="$LOCK" \
    "$SCRIPT" < /dev/null >> "$LOG" 2>&1 &
  disown
  exit 0
}
trap 'hand_off "SIGTERM received"' TERM

echo "[$(date '+%F %T')] watchdog started (pid $$); interval ${INTERVAL}s; self-restart ${SELF_RESTART_HOURS}h"

while true; do
  IFS='|' read -r -a entries <<< "$WATCH"
  all_done=1
  for entry in "${entries[@]}"; do
    dataset="${entry%%:*}"
    rest="${entry#*:}"
    gpu="${rest%%:*}"
    rest="${rest#*:}"
    seeds="${rest%%:*}"
    extra="${rest#*:}"
    workspace="rdbdiff_workspace/${dataset}/paper_repro"

    # Stop watching once every requested seed has finished.
    done_all=1
    for seed in $seeds; do
      [ -f "${workspace}/seed_${seed}/synthetic_data_sampled_structure_size_multiplier_1.0/eval_result.pkl" ] || done_all=0
    done
    [ "$done_all" -eq 1 ] && continue
    all_done=0

    # 9>&- is load-bearing: without it, this command-substitution subshell inherits fd 9,
    # execs into resume_runs.sh, and resume_runs.sh's `setsid nohup ... & disown` training
    # jobs inherit it too -- keeping the flock held for their entire ~20h lifetime, long
    # after this watchdog process exits. That is exactly what happened on the first run:
    # the 12h self-restart handoff found the lock still "held" by six training processes
    # that were not watchdogs at all, gave up silently, and nothing watched the runs for
    # the next ~9.5h (no work was lost -- each job's own --max_hours deadline still saved
    # a checkpoint and exited cleanly -- but nothing relaunched them afterward).
    out=$(RDBDIFF_DATASET="$dataset" RDBDIFF_GPUS="$gpu" RDBDIFF_EXTRA_ARGS="$extra" \
      scripts/resume_runs.sh $seeds 2>&1 9>&-)
    # Only log when it actually acted; "already running" every 10 minutes is noise.
    if grep -qE "resuming from checkpoint|starting fresh" <<< "$out"; then
      echo "[$(date '+%F %T')] ${dataset} seed(s) ${seeds}: relaunched on GPU ${gpu}"
      grep -E "seed [0-9]+:" <<< "$out" | sed 's/^/    /'
    fi
  done

  if [ "$all_done" -eq 1 ]; then
    echo "[$(date '+%F %T')] every watched seed has finished; watchdog exiting"
    exit 0
  fi

  # Retire before the node's ~24h SIGTERM can reach us, so the successor is always young.
  if [ $(( $(date +%s) - START_TS )) -ge $(( SELF_RESTART_HOURS * 3600 )) ]; then
    hand_off "reached ${SELF_RESTART_HOURS}h"
  fi

  # Backgrounded sleep + wait, so a trapped SIGTERM is handled immediately. bash defers
  # traps until the running foreground command finishes, and a foreground `sleep 600` would
  # delay the hand-off by up to the full interval.
  # 9>&- so the sleep child does not inherit the lock fd. If it did, it would keep the
  # flock held after this process exits, and the successor spawned by hand_off would be
  # refused the lock and die -- leaving nothing watching the runs.
  sleep "$INTERVAL" 9>&- &
  SLEEP_PID=$!
  wait $SLEEP_PID || true
done
