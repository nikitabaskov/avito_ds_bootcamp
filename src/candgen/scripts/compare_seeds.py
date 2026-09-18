import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from candgen.core.evaluation import average_seed_metrics, compare_per_query
from candgen.scripts.common import EXPERIMENTS_DIR


def load_runs(names: list[str]) -> dict[int, tuple[str, pl.DataFrame]]:
    runs = {}
    for name in names:
        path = EXPERIMENTS_DIR / name
        report = json.loads((path / "report.json").read_text())
        seed = (report.get("ranker") or report["model"]["ranker"])["random_seed"]
        if seed in runs:
            raise ValueError(f"duplicate seed {seed}")
        runs[seed] = (name, pl.read_parquet(path / "per_query.parquet"))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates, baselines = load_runs(args.candidate), load_runs(args.baseline)
    if candidates.keys() != baselines.keys():
        parser.error("candidate and baseline must have the same seeds")
    seeds = sorted(candidates)
    candidate = average_seed_metrics([candidates[s][1] for s in seeds])
    baseline = average_seed_metrics([baselines[s][1] for s in seeds])
    per_seed = [
        {
            "seed": s,
            "candidate": candidates[s][0],
            "baseline": baselines[s][0],
            "candidate_recall@50": float(candidates[s][1]["recall"].mean()),
            "baseline_recall@50": float(baselines[s][1]["recall"].mean()),
        }
        for s in seeds
    ]
    report = {
        "method": "mean metric per query across matched training seeds; paired query bootstrap",
        "limitation": "conditional on these seeds and this dev split; not ensemble predictions",
        "seeds": per_seed,
        "candidate_recall@50": float(candidate["recall"].mean()),
        "candidate_seed_sd": float(np.std([r["candidate_recall@50"] for r in per_seed], ddof=1))
        if len(seeds) > 1
        else None,
        "comparison": compare_per_query(candidate, baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
