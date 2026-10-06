"""Which (variant, task) pairs have been scored, and which are still missing.

`run.py --dataset X` scores one variant at a time and each Berka run takes days, so the set of
things that *could* be scored drifts as runs land. This prints the whole matrix: seeds that have
sampled data on disk, tasks already scored, and the gaps.

    conda run -n tabsyn python -m downstream.coverage
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT
from downstream.run import DATASETS


def sampled_seeds(cfg):
    variant = cfg.get("variant", "")
    pattern = os.path.join(
        REPO_ROOT, "rdbdiff_workspace", cfg["grdm_dataset"], "paper_repro", "seed_*",
        f"synthetic_data_sampled_structure_size_multiplier_1.0{variant}", "synthetic_database",
    )
    seeds = []
    for path in glob.glob(pattern):
        seed_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seeds.append(int(seed_dir.split("_")[1]))
    return sorted(seeds)


def scored_seeds(name, task):
    path = os.path.join(REPO_ROOT, "downstream", "results", f"{name}__{task}.json")
    if not os.path.exists(path):
        return None
    records = json.load(open(path))["records"]
    return sorted({r["grdm_seed"] for r in records if r["grdm_seed"] is not None})


def main():
    rows, missing = [], []
    for name, cfg in DATASETS.items():
        seeds = sampled_seeds(cfg)
        cells = []
        for task in cfg["tasks"]:
            scored = scored_seeds(name, task)
            short = task.replace("california_", "").replace("berka_", "")
            if scored is None:
                cells.append(f"{short}(-)")
                if seeds:
                    missing.append((name, task, seeds))
            else:
                cells.append(f"{short}({len(scored)})")
                todo = sorted(set(seeds) - set(scored))
                if todo:
                    missing.append((name, task, todo))
        rows.append((name, seeds, cells))

    print(f"{'variant':22s} {'sampled seeds':16s} tasks scored (n seeds)")
    print("-" * 78)
    for name, seeds, cells in rows:
        print(f"{name:22s} {str(seeds):16s} {', '.join(cells)}")

    print()
    if not missing:
        print("Every variant with sampled data has been scored on all of its tasks.")
        return
    print("Scoreable but not yet scored:")
    for name, task, seeds in missing:
        print(f"  {name} / {task}  seeds {seeds}")
        print(f"    conda run --no-capture-output -n tabsyn python -m downstream.run "
              f"--dataset {name} --tasks {task} --grdm-seeds {' '.join(map(str, seeds))}")


if __name__ == "__main__":
    main()
