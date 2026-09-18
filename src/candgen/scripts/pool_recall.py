import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from candgen.core.data import SPLIT_DIR
from candgen.core.dense import DenseConfig
from candgen.core.evaluation import paired_bootstrap, pool_rows
from candgen.core.features import ItemTable, candidate_pool
from candgen.core.history import full_view, history_pairs, query_centers
from candgen.core.metrics import recall_at_k
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.experiment import fixed_valid_texts
from candgen.scripts.predict import git_state
from candgen.scripts.runs import RetrievalConfig, fuse, retrieve, rows_to_ids

VARIANTS = {"filters": True, "no_filters": False}


def evaluate(
    config: RetrievalConfig,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    centers: np.ndarray | None,
    timings: dict,
) -> dict[str, np.ndarray]:
    item_ids = corpus["item_id"].to_list()
    relevant = queries["item_ids"].to_list()
    runs = retrieve(config, "split", corpus, queries, "dev", timings, centers)
    depths = {name: config.list_depth(name) for name in config.list_names()}
    pool = rows_to_ids(
        item_ids, pool_rows(candidate_pool(runs, depths, config.rrf_k), len(relevant))
    )
    out = {
        "pool": recall_at_k(pool, relevant, max(map(len, pool))),
        "rrf@50": recall_at_k(rows_to_ids(item_ids, fuse(config, runs, timings)), relevant, 50),
        "pool_size": np.array([len(p) for p in pool], dtype=np.float64),
    }
    for name in ("dense_global", "dense_local", "dense_radius"):
        if name in depths:
            rows = runs[name][0][:, : depths[name]]
            out[name] = recall_at_k(rows_to_ids(item_ids, rows), relevant, depths[name])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-k", type=int, default=400)
    parser.add_argument("--local-k", type=int, default=300)
    parser.add_argument("--radius-km", type=float, default=25.0)
    parser.add_argument("--radius-k", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENTS_DIR / "EXP-010" / "pool_a" / "report.json"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    timings: dict[str, float] = {}
    corpus, queries, _ = load_eval("dev")
    centers = None
    if args.radius_k:
        train_contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
        valid_texts = fixed_valid_texts(train_contexts)
        pairs = history_pairs(
            train_contexts.filter(~pl.col("query_text").is_in(valid_texts.implode())), corpus
        )
        centers = query_centers(full_view(queries, pairs), ItemTable(corpus))
    results, configs = {}, {}
    for name, filters in VARIANTS.items():
        config = RetrievalConfig(
            global_k=args.global_k,
            local_k=args.local_k,
            radius_km=args.radius_km,
            radius_k=args.radius_k,
            dense_config=DenseConfig(query_filters=filters),
        )
        configs[name] = dataclasses.asdict(config)
        results[name] = evaluate(config, corpus, queries, centers, timings)

    masks = {
        "all": np.ones(queries.height, dtype=bool),
        "filters_set": (queries["search_infm_params_text"] != "").to_numpy(),
        "filters_empty": (queries["search_infm_params_text"] == "").to_numpy(),
    }
    base, candidate = results["filters"], results["no_filters"]
    report = {
        "experiment": "EXP-010/pool_a",
        "git": git_state(),
        "queries": queries.height,
        "configs": configs,
        "metrics": {
            name: {m: float(v.mean()) for m, v in r.items()} for name, r in results.items()
        },
        "no_filters_vs_filters": {
            slice_name: {
                "n": int(mask.sum()),
                **{
                    metric: paired_bootstrap(candidate[metric][mask], base[metric][mask])
                    for metric in ("pool", "rrf@50", "dense_global")
                },
            }
            for slice_name, mask in masks.items()
        },
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
