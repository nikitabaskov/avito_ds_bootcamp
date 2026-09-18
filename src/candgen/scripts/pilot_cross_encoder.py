import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from catboost import CatBoostRanker

from candgen.core.crossenc import CrossEncoderConfig
from candgen.core.data import SPLIT_DIR
from candgen.core.dense import default_device
from candgen.core.evaluation import paired_bootstrap
from candgen.core.features import ItemTable
from candgen.core.history import add_history_features, full_view, history_pairs, query_centers
from candgen.core.metrics import recall_at_k
from candgen.core.ranker import make_pool, select_top
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.experiment import fixed_valid_texts
from candgen.scripts.predict import git_state, model_config
from candgen.scripts.runs import (
    cross_encoder_features,
    load_corpus_embeddings,
    microcat_features,
    microcat_index,
    pool_features,
    retrieve,
    rows_to_ids,
)

PARENT = EXPERIMENTS_DIR / "EXP-009" / "pool400x300_s42" / "model.cbm"
BLEND_WEIGHTS = (0.5, 1.0)
RRF_K = 60


def recall(frame: pl.DataFrame, scores: np.ndarray, queries: pl.DataFrame, ids: list[str]):
    rows = select_top(frame, scores, queries.height)
    return recall_at_k(rows_to_ids(ids, rows), queries["item_ids"].to_list(), 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=PARENT)
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=CrossEncoderConfig.top_k)
    parser.add_argument("--batch-size", type=int, default=CrossEncoderConfig.batch_size)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_DIR / "EXP-012" / "pilot")
    args = parser.parse_args()

    started = time.perf_counter()
    timings: dict[str, float] = {}
    meta = json.loads(args.model.with_suffix(".json").read_text())
    retrieval = model_config(meta)
    geo = meta["history"]["geo_history"]
    transitions = meta["history"]["transitions"]
    alpha = meta["history"].get("transition_alpha") or 0.0
    ce_config = CrossEncoderConfig(top_k=args.top_k, batch_size=args.batch_size)

    corpus, dev_queries, _ = load_eval("dev")
    item_ids = corpus["item_id"].to_list()
    train_contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
    valid_texts = fixed_valid_texts(train_contexts)
    pairs = history_pairs(
        train_contexts.filter(~pl.col("query_text").is_in(valid_texts.implode())), corpus
    )
    del train_contexts
    items = ItemTable(corpus)
    views = full_view(dev_queries, pairs)
    run_key = "dev"
    runs = retrieve(
        retrieval, "split", corpus, dev_queries, run_key, timings, query_centers(views, items)
    )
    item_vectors = torch.from_numpy(
        load_corpus_embeddings(retrieval.dense_config, "split", corpus, timings)
    ).to(default_device())
    frame = pool_features(retrieval, runs, dev_queries, run_key, items, item_vectors, timings)
    del item_vectors
    frame = add_history_features(frame, views, items, geo, transitions, alpha)
    index = microcat_index(retrieval.dense_config, pairs, timings)
    frame = microcat_features(
        retrieval.dense_config, index, frame, views, dev_queries, corpus, run_key, timings
    )
    del index, pairs
    torch.cuda.empty_cache()
    if args.queries:
        dev_queries = dev_queries.head(args.queries)
        frame = frame.filter(pl.col("q") < args.queries)

    model = CatBoostRanker()
    model.load_model(str(args.model))
    base_scores = model.predict(make_pool(frame, meta["features"]))
    frame = cross_encoder_features(ce_config, frame, dev_queries, corpus, run_key, timings)

    model_rank = (
        frame.select("q", score=pl.Series(base_scores))
        .select(pl.col("score").rank("ordinal", descending=True).over("q"))
        .to_series()
        .to_numpy()
    )
    ce_rank = frame["ce_rank"].fill_null(np.inf).to_numpy()
    ce_only = np.where(np.isfinite(ce_rank), -ce_rank, -np.inf)
    base = recall(frame, base_scores, dev_queries, item_ids)
    variants = {"cross_encoder_only": ce_only}
    for w in BLEND_WEIGHTS:
        variants[f"blend_w{w}"] = 1 / (RRF_K + model_rank) + w / (RRF_K + ce_rank)
    results = {}
    for name, scores in variants.items():
        r = recall(frame, scores, dev_queries, item_ids)
        results[name] = {"recall@50": float(r.mean()), **paired_bootstrap(r, base)}

    positives = (
        dev_queries.select(q=pl.int_range(0, pl.len(), dtype=pl.Int32), item_id="item_ids")
        .explode("item_id")
        .join(pl.DataFrame({"item_id": item_ids}).with_row_index("row"), on="item_id")
        .with_columns(pl.col("row").cast(frame["row"].dtype))
        .join(
            frame.select("q", "row", "rrf_rank", "ce_rank", model_rank=pl.Series(model_rank)),
            on=["q", "row"],
        )
    )
    lost = positives.filter(pl.col("model_rank") > 50)
    report = {
        "experiment": "EXP-012/pilot",
        "parent_model": str(args.model),
        "git": git_state(),
        "cross_encoder": dataclasses.asdict(ce_config),
        "queries": dev_queries.height,
        "scored_pairs": int(frame["ce_score"].is_not_null().sum()),
        "parent_recall@50": float(base.mean()),
        "variants": results,
        "selection_lost_positives": {
            "n": lost.height,
            "scored": int(lost["ce_rank"].is_not_null().sum()),
            "ce_rank_le_50": int((lost["ce_rank"] <= 50).sum()),
        },
        "found_positives_ce_rank_le_50": {
            "n": positives.filter(pl.col("model_rank") <= 50).height,
            "ce_rank_le_50": int(
                positives.filter((pl.col("model_rank") <= 50) & (pl.col("ce_rank") <= 50)).height
            ),
        },
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (
        args.output / f"report{'' if args.queries is None else f'_head{args.queries}'}.json"
    ).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
