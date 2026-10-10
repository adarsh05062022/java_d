#!/usr/bin/env bash
# Commit and push the paper_repro fidelity results (scores and logs of finished seeds) to origin.
# Checkpoints and synthetic databases stay in rdbdiff_workspace/ and are never pushed.
#
# Safe to re-run at any time: results/paper_repro/ only changes when a seed finishes, so this
# commits nothing otherwise. Only that folder is committed, whatever else is staged.
#
#   scripts/publish_repro_results.sh                     # berka and california
#   scripts/publish_repro_results.sh berka california ccs

set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root (this script lives in scripts/)

status=$(python3 scripts/repro_summary.py "$@")
echo "finished seeds: ${status}"

git add results/paper_repro
if git diff --cached --quiet -- results/paper_repro; then
  echo "no new results to publish"
  exit 0
fi

git commit -q -m "paper_repro results: ${status} seeds finished" -- results/paper_repro
git push -q origin HEAD
echo "pushed $(git rev-parse --short HEAD) to origin/$(git rev-parse --abbrev-ref HEAD)"
