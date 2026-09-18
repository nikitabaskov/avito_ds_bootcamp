import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRanker

from candgen.core.data import SEED, SPLIT_DIR, repeated_text_queries, sample_eval_queries
from candgen.core.evaluation import paired_bootstrap
from candgen.core.features import ItemTable
from candgen.core.history import history_pairs
from candgen.core.metrics import recall_at_k
from candgen.core.ranker import make_pool, select_top
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.experiment import fixed_valid_texts
from candgen.scripts.predict import (
    expected_features,
    feature_frame,
    feature_spec,
    git_state,
    model_config,
)
from candgen.scripts.runs import rows_to_ids

MODEL = EXPERIMENTS_DIR / "EXP-009" / "pool400x300_s42" / "model.cbm"
QUERIES = 2452


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--queries", type=int, default=QUERIES)
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENTS_DIR / "repeated_text" / "exp014_exact"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    timings: dict[str, float] = {}
    meta = json.loads(args.model.with_suffix(".json").read_text())
    config, spec = model_config(meta), feature_spec(meta)
    if meta["features"] != expected_features(spec, config) or spec["microcats"] != "neighbors":
        raise SystemExit(f"{args.model} must be a neighbors-microcat model")

    corpus, _, _ = load_eval("dev")
    item_ids = corpus["item_id"].to_list()
    contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
    valid_texts = fixed_valid_texts(contexts)
    train_texts = sample_eval_queries(contexts, meta["train_queries"], SEED)["query_text"]
    queries = repeated_text_queries(contexts, pl.concat([valid_texts, train_texts]), args.queries)
    history = contexts.filter(~pl.col("query_text").is_in(valid_texts.implode())).join(
        queries.select("query_id"), on="query_id", how="anti"
    )
    pairs = history_pairs(history, corpus)
    items = ItemTable(corpus)
    run_key = f"repeated{args.queries}"
    relevant = queries["item_ids"].to_list()

    model = CatBoostRanker()
    model.load_model(str(args.model))
    recalls, exact_flag, pool = {}, None, None
    for name, exact_prior in (("neighbors", False), ("exact", True)):
        frame = feature_frame(
            spec, config, "split", corpus, queries, run_key, items, pairs, timings, exact_prior
        )
        scores = model.predict(make_pool(frame, meta["features"]))
        rows = select_top(frame, scores, queries.height)
        recalls[name] = recall_at_k(rows_to_ids(item_ids, rows), relevant, 50)
        if exact_prior:
            exact_flag = (frame.group_by("q").agg(pl.col("mc_is_exact_match").first()).sort("q"))[
                "mc_is_exact_match"
            ].to_numpy() == 1.0
        else:
            full = select_top(frame, scores, queries.height, k=np.iinfo(np.int32).max)
            pool = recall_at_k(rows_to_ids(item_ids, full), relevant, np.iinfo(np.int32).max)
        del frame

    has_center = (
        queries["search_location_id"]
        .is_in(items.locations["item_location_id"].implode())
        .to_numpy()
    )
    filters_set = (queries["search_infm_params_text"] != "").to_numpy()
    masks = {
        "all": np.ones(queries.height, dtype=bool),
        "exact_match": exact_flag,
        "no_exact_match": ~exact_flag,
        "filters_set": filters_set,
        "filters_empty": ~filters_set,
        "has_center": has_center,
        "no_center": ~has_center,
    }
    report = {
        "experiment": "EXP-014/repeated_text",
        "model": str(args.model),
        "git": git_state(),
        "protocol": (
            "held-out train contexts of texts outside the model's train sample and ranker valid "
            "texts, with at least one other context of the same text left in history; history "
            "excludes only the held-out contexts"
        ),
        "queries": queries.height,
        "history_contexts": history.height,
        "pool_recall": float(pool.mean()),
        "recall@50": {name: float(r.mean()) for name, r in recalls.items()},
        "exact_vs_neighbors": {
            name: {
                "n": int(mask.sum()),
                "neighbors_recall@50": float(recalls["neighbors"][mask].mean()),
                "exact_recall@50": float(recalls["exact"][mask].mean()),
                **paired_bootstrap(recalls["exact"][mask], recalls["neighbors"][mask]),
            }
            for name, mask in masks.items()
            if mask.any()
        },
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "query_id": queries["query_id"],
            "exact_match": exact_flag,
            "recall_neighbors": recalls["neighbors"],
            "recall_exact": recalls["exact"],
            "pool_recall": pool,
        }
    ).write_parquet(args.output / "per_query.parquet")
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (args.output / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
