import argparse
import dataclasses
import json
import time

import numpy as np
import polars as pl
import torch

from candgen.core.data import (
    ARTIFACTS_DIR,
    SEED,
    SPLIT_DIR,
    load_corpus,
    sample_eval_queries,
    stable_hash,
)
from candgen.core.dense import default_device
from candgen.core.evaluation import paired_bootstrap, recall_report
from candgen.core.features import FEATURES, ItemTable, attach_labels
from candgen.core.metrics import recall_at_k
from candgen.core.ranker import RankerConfig, make_pool, select_top, train_ranker
from candgen.scripts.common import peak_rss_gb, write_report
from candgen.scripts.runs import (
    RetrievalConfig,
    load_corpus_embeddings,
    pool_features,
    retrieve,
    rows_to_ids,
)

MODELS_DIR = ARTIFACTS_DIR / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-queries", type=int, default=6000)
    parser.add_argument("--valid-share", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=RankerConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=RankerConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=RankerConfig.depth)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default=RankerConfig.task_type)
    args = parser.parse_args()
    retrieval = RetrievalConfig()
    ranker = RankerConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        task_type=args.task_type,
    )
    tag = f"n{args.train_queries}_d{ranker.depth}_lr{ranker.learning_rate}"

    corpus = load_corpus("split")
    item_ids = corpus["item_id"].to_list()
    train_contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
    train_queries = sample_eval_queries(train_contexts, args.train_queries, SEED)
    dev_queries = pl.read_parquet(SPLIT_DIR / "eval_dev.parquet")
    seen_items = set(train_contexts["item_ids"].explode(empty_as_null=True).unique())
    del train_contexts

    timings: dict[str, float] = {}
    train_key = f"train{args.train_queries}"
    train_runs = retrieve(retrieval, "split", corpus, train_queries, train_key, timings)
    dev_runs = retrieve(retrieval, "split", corpus, dev_queries, "dev", timings)

    t = time.perf_counter()
    items = ItemTable(corpus)
    embeddings = load_corpus_embeddings(retrieval.dense_config, "split", corpus, timings)
    item_vectors = torch.from_numpy(embeddings).to(default_device())
    del embeddings
    timings["item_table_s"] = time.perf_counter() - t

    train_frame = attach_labels(
        pool_features(
            retrieval, train_runs, train_queries, train_key, items, item_vectors, timings
        ),
        train_queries["item_ids"].to_list(),
        item_ids,
    )
    dev_frame = pool_features(retrieval, dev_runs, dev_queries, "dev", items, item_vectors, timings)
    del items, item_vectors

    valid_bucket = (
        pl.Series(
            stable_hash((f"valid\x1f{t}" for t in train_queries["query_text"]), SEED),
            dtype=pl.UInt64,
        )
        % 1000
    )
    is_valid = pl.DataFrame(
        {
            "q": np.arange(train_queries.height, dtype=np.int32),
            "valid": valid_bucket < 1000 * args.valid_share,
        }
    )
    with_positive = (
        train_frame.group_by("q").agg(pl.col("label").max() > 0).filter("label").select("q")
    )
    train_frame = (
        train_frame.join(with_positive, on="q", how="semi")
        .join(is_valid, on="q")
        .sort("q", "rrf_rank")
    )
    fit_frame = train_frame.filter(~pl.col("valid"))
    valid_frame = train_frame.filter(pl.col("valid"))

    t = time.perf_counter()
    model = train_ranker(fit_frame, valid_frame, ranker)
    timings["train_s"] = time.perf_counter() - t

    t = time.perf_counter()
    scores = model.predict(make_pool(dev_frame))
    timings["dev_predict_s"] = time.perf_counter() - t
    relevant = dev_queries["item_ids"].to_list()
    selected = rows_to_ids(item_ids, select_top(dev_frame, scores, dev_queries.height))
    rrf_selected = rows_to_ids(
        item_ids, select_top(dev_frame, -dev_frame["rrf_rank"].to_numpy(), dev_queries.height)
    )
    pools = dev_frame.group_by("q", maintain_order=True).agg("row")
    pool_rows = [np.empty(0, dtype=np.int64) for _ in range(dev_queries.height)]
    for q, rows in pools.iter_rows():
        pool_rows[q] = np.array(rows, dtype=np.int64)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"ranker_{tag}.cbm"
    model.save_model(str(model_path))
    model_meta = {
        "features": FEATURES,
        "retrieval": dataclasses.asdict(retrieval),
        "ranker": dataclasses.asdict(ranker),
        "best_iteration": model.get_best_iteration(),
        "train_queries": args.train_queries,
    }
    model_path.with_suffix(".json").write_text(json.dumps(model_meta, indent=2, ensure_ascii=False))

    report = {
        "method": "catboost",
        "part": "dev",
        "model": str(model_path),
        **model_meta,
        "train": {
            "sampled_queries": train_queries.height,
            "queries_with_positive_in_pool": with_positive.height,
            "fit_queries": fit_frame["q"].n_unique(),
            "valid_queries": valid_frame["q"].n_unique(),
            "fit_rows": fit_frame.height,
            "positive_rows": int(fit_frame["label"].sum()),
            "best_valid": model.get_best_score().get("validation"),
        },
        **recall_report(dev_queries, selected, seen_items),
        "pool_recall": float(recall_at_k(rows_to_ids(item_ids, pool_rows), relevant, 10**6).mean()),
        "rrf_same_pool": float(recall_at_k(rrf_selected, relevant, 50).mean()),
        "vs_rrf": paired_bootstrap(
            recall_at_k(selected, relevant, 50), recall_at_k(rrf_selected, relevant, 50)
        ),
        "feature_importance": dict(
            sorted(
                zip(
                    FEATURES,
                    map(float, model.get_feature_importance(type="PredictionValuesChange")),
                    strict=True,
                ),
                key=lambda kv: -kv[1],
            )
        ),
        "timings": timings,
        "peak_rss_gb": peak_rss_gb(),
    }
    write_report(f"catboost_dev_{tag}", report)


if __name__ == "__main__":
    main()
