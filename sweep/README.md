# Berka-utility sweep — unattended runner

Plan: `~/.claude/plans/run-all-i-have-soft-lighthouse.md`.

## Run it

```bash
cd /scratch/s25017/ClavaDDPM/rdb-diffusion
setsid nohup ./sweep/run_sweep.sh < /dev/null >> sweep/run_sweep.log 2>&1 & disown
```

It keeps running across logout and the node's ~24 h SIGTERM (flock + 12 h self-hand-off).

## Watch it

```bash
cat sweep/status.md          # per-variant table: seeds trained / scored / retained %
tail -f sweep/run_sweep.log  # scheduler log
kill -0 "$(cat sweep/.pid)" && echo alive || echo stopped
```

`sweep/status.md` ends with `## SWEEP COMPLETE` when everything is scored;
`docs/downstream/RESULTS_downstream.md` is regenerated and `sweep/coverage.txt` written at that point.

## Stop it

```bash
kill "$(cat sweep/.pid)"      # NOTE: this hands off to a fresh copy (by design).
```

To stop it for good: `rm sweep/manifest.tsv` first (loop exits when nothing is pending and
everything present is scored), or `flock -n sweep/.lock true` after killing to confirm the
lock is free, then don't relaunch.

## What runs

`sweep/manifest.tsv` — one variant per line: `name <TAB> seeds <TAB> kind <TAB> flags`.
`kind`: `berka` | `berka_reparam` (decode before scoring) | `california`.
Add the two discrete rows after the discrete-diffusion smoke test:

```
berka_split_discrete	0,1,2	berka	--categorical_diffusion multinomial
california_discrete	0,1,2	california	--categorical_diffusion multinomial
```

The runner re-reads the manifest every tick, so appending rows mid-run is fine.

## Knobs (env vars at launch)

| var | default | meaning |
|---|---|---|
| `SWEEP_INTERVAL` | 300 | seconds between scheduler ticks |
| `SWEEP_GPU_MEM_MAX_MIB` | 120000 | a GPU is usable if its used memory is below this (raise to co-locate on busier GPUs, lower to only take idle ones) |
| `SWEEP_RUNS_PER_GPU` | 3 | co-locate up to this many sweep runs on one GPU (each GRDM run is ~1.5 GB; contention ~1.2–1.5x per run) |
| `SWEEP_MAX_CONCURRENT` | 24 | global cap on simultaneous training runs |
| `SWEEP_MAX_HOURS` | 20 | per-run wall budget before checkpoint+resume |
| `SWEEP_GPUS` | all | restrict to e.g. `"2 3 5"` |
| `SWEEP_SELF_RESTART_HOURS` | 12 | hand-off cadence |

To change these on a running sweep: `kill -9 $(cat sweep/.pid)`, kill any handoff survivor, then relaunch with the new env (the hand-off passes the *current* values forward, so a plain `kill` keeps the old ones).

## How it works each tick

1. Find usable GPUs (`nvidia-smi` mem, minus GPUs already hosting a sweep run), launch
   pending `(variant, seed)` via `scripts/resume_runs.sh` (which owns setsid/nohup/checkpoint/max_hours).
2. For any variant whose every seed has `…/eval_result.pkl`: decode (reparam only) →
   `downstream.run` → `dep_fidelity.run` (Berka) → write `sweep/state/<name>.scored`.
3. Rewrite `sweep/status.md`.
4. When all manifest variants are scored → `downstream.report` + `downstream.coverage`, exit.

Idempotent and stateless — a missed hand-off is recovered by just re-running the launch line.
