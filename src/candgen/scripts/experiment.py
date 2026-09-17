import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from catboost import CatBoostRanker

from candgen.core.data import SEED, SPLIT_DIR, sample_eval_queries, stable_hash
from candgen.core.dense import default_device
from candgen.core.diagnostics import error_map, positive_outcomes, query_outcomes
from candgen.core.evaluation import paired_bootstrap, recall_report
from candgen.core.features import FEATURES, ItemTable, attach_labels
from candgen.core.history import (
    FOLDS,
    GEO_MODES,
    TRANSITION_MODES,
    add_history_features,
    crossfit_views,
    full_view,
    history_features,
    history_pairs,
    location_centers,
)
from candgen.core.ranker import RankerConfig, make_pool, select_top, train_ranker
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.predict import git_state
from candgen.scripts.runs import (
    RetrievalConfig,
    load_corpus_embeddings,
    pool_features,
    retrieve,
    rows_to_ids,
)

PROTOCOL = "v1"
B0 = "EXP-000/b0"
VALID_BASE_QUERIES = 6000
VALID_SHARE = 0.1
VALID_TEXTS_PATH = SPLIT_DIR / "ranker_valid_texts.parquet"


def valid_bucket(texts: pl.Series) -> pl.Series:
    return (
        pl.Series(stable_hash((f"valid\x1f{t}" for t in texts), SEED), dtype=pl.UInt64) % 1000
        < 1000 * VALID_SHARE
    )


def fixed_valid_texts(train_contexts: pl.DataFrame) -> pl.Series:
    if not VALID_TEXTS_PATH.exists():
        base = sample_eval_queries(train_contexts, VALID_BASE_QUERIES, SEED)
        base.filter(valid_bucket(base["query_text"])).select("query_text").sort(
            "query_text"
        ).write_parquet(VALID_TEXTS_PATH)
    return pl.read_parquet(VALID_TEXTS_PATH)["query_text"]


def train_model(
    args: argparse.Namespace,
    retrieval: RetrievalConfig,
    ranker: RankerConfig,
    features: list[str],
    corpus: pl.DataFrame,
    train_contexts: pl.DataFrame,
    pairs: pl.DataFrame,
    items: ItemTable,
    item_vectors: torch.Tensor,
    timings: dict,
) -> tuple[CatBoostRanker, dict]:
    train_queries = sample_eval_queries(train_contexts, args.train_queries, SEED)
    valid_texts = fixed_valid_texts(train_contexts)
    train_key = f"train{args.train_queries}"
    runs = retrieve(retrieval, "split", corpus, train_queries, train_key, timings)
    roles = train_queries.select(
        q=pl.int_range(0, pl.len(), dtype=pl.Int32),
        valid=pl.col("query_text").is_in(valid_texts.implode()),
        excluded=valid_bucket(train_queries["query_text"])
        & ~pl.col("query_text").is_in(valid_texts.implode()),
    )
    frame = pool_features(retrieval, runs, train_queries, train_key, items, item_vectors, timings)
    t = time.perf_counter()
    views = crossfit_views(train_queries, pairs, roles.filter("valid").select("q"))
    frame = attach_labels(
        add_history_features(
            frame, views, items, args.geo_history, args.transitions, args.transition_alpha
        ),
        train_queries["item_ids"].to_list(),
        corpus["item_id"].to_list(),
    )
    timings["train_history_s"] = time.perf_counter() - t
    with_positive = frame.group_by("q").agg(pl.col("label").max() > 0).filter("label").select("q")
    frame = (
        frame.join(with_positive, on="q", how="semi")
        .join(roles, on="q")
        .filter(~pl.col("excluded"))
        .sort("q", "rrf_rank")
    )
    fit_frame = frame.filter(~pl.col("valid"))
    valid_frame = frame.filter(pl.col("valid"))

    t = time.perf_counter()
    model = train_ranker(fit_frame, valid_frame, ranker, features)
    timings["train_s"] = time.perf_counter() - t
    return model, {
        "sampled_queries": train_queries.height,
        "queries_with_positive_in_pool": with_positive.height,
        "excluded_valid_bucket_queries": int(roles["excluded"].sum()),
        "fit_queries": fit_frame["q"].n_unique(),
        "valid_queries": valid_frame["q"].n_unique(),
        "valid_texts_file": str(VALID_TEXTS_PATH),
        "fit_rows": fit_frame.height,
        "positive_rows": int(fit_frame["label"].sum()),
        "best_iteration": model.get_best_iteration(),
        "best_valid": model.get_best_score().get("validation"),
    }


def compare(per_query: pl.DataFrame, reference: str) -> dict | None:
    path = EXPERIMENTS_DIR / reference / "per_query.parquet"
    if not path.exists():
        return None
    ref = pl.read_parquet(path).select("query_id", ref_recall="recall")
    joined = per_query.select("query_id", "recall").join(ref, on="query_id", how="full")
    if joined["query_id"].null_count() or joined["query_id_right"].null_count():
        raise SystemExit(f"{reference} was evaluated on different queries")
    return {
        "reference": reference,
        "reference_recall@50": float(joined["ref_recall"].mean()),
        **paired_bootstrap(joined["recall"].to_numpy(), joined["ref_recall"].to_numpy()),
    }


def error_examples(
    positives: pl.DataFrame,
    per_query: pl.DataFrame,
    queries: pl.DataFrame,
    selected: list[np.ndarray],
    corpus: pl.DataFrame,
    n: int,
) -> list[dict]:
    titles = corpus["item_title_raw"].to_list()
    locations = corpus["item_location_id"].to_list()
    failed = per_query.filter(pl.col("recall") < 1)
    order = np.argsort(np.array(stable_hash(failed["query_id"], SEED), dtype=np.uint64))
    examples = []
    for q in failed["q"].to_numpy()[order[:n]].tolist():
        query = queries.row(q, named=True)
        lost = positives.filter((pl.col("q") == q) & ~pl.col("in_top50"))
        examples.append(
            {
                "query_id": query["query_id"],
                "query": query["query_text"],
                "filters": query["search_infm_params_text"],
                "location": query["search_location_id"],
                "category": query["search_category"],
                "n_pos": per_query["n_pos"][q],
                "recall": per_query["recall"][q],
                "lost": [
                    {
                        "title": titles[r["row"]],
                        "location": r["item_location_id"],
                        "model_rank": r["model_rank"],
                        "n_lists": r["n_lists"],
                        "dist_km": r["dist_km"],
                        "filter_overlap": r["filter_overlap"],
                    }
                    for r in lost.iter_rows(named=True)
                ],
                "selected_top5": [
                    {"title": titles[r], "location": locations[r]} for r in selected[q][:5].tolist()
                ],
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--parent", default=B0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--drop-features", nargs="*", default=[])
    parser.add_argument("--geo-history", choices=GEO_MODES, default="none")
    parser.add_argument("--transitions", choices=TRANSITION_MODES, default="none")
    parser.add_argument("--transition-alpha", type=float, default=10.0)
    parser.add_argument("--train-queries", type=int, default=VALID_BASE_QUERIES)
    parser.add_argument("--iterations", type=int, default=RankerConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=RankerConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=RankerConfig.depth)
    parser.add_argument("--seed", type=int, default=RankerConfig.random_seed)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default=RankerConfig.task_type)
    parser.add_argument("--examples", type=int, default=30)
    args = parser.parse_args()

    name = f"{args.exp}/{args.variant}"
    out_dir = EXPERIMENTS_DIR / name
    if (out_dir / "report.json").exists():
        raise SystemExit(f"{out_dir} already has a report")
    available = [*FEATURES, *history_features(args.geo_history, args.transitions)]
    unknown = set(args.drop_features) - set(available)
    if unknown:
        parser.error(f"unknown features: {sorted(unknown)}")
    features = [f for f in available if f not in args.drop_features]
    retrieval = RetrievalConfig()
    ranker = RankerConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.seed,
        task_type=args.task_type,
    )

    started = time.perf_counter()
    timings: dict[str, float] = {}
    corpus, dev_queries, seen_items = load_eval("dev")
    item_ids = corpus["item_id"].to_list()
    train_contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
    valid_texts = fixed_valid_texts(train_contexts)
    t = time.perf_counter()
    history = train_contexts.filter(~pl.col("query_text").is_in(valid_texts.implode()))
    pairs = history_pairs(history, corpus)
    dev_centers = dev_queries.join(location_centers(pairs), on="search_location_id", how="left")
    timings["history_s"] = time.perf_counter() - t
    history_info = {
        "source": "contexts_train without ranker valid texts",
        "texts": history["query_text"].n_unique(),
        "contexts": history.height,
        "pairs_with_coords": pairs.height,
        "train_folds": FOLDS,
        "geo_history": args.geo_history,
        "transitions": args.transitions,
        "transition_alpha": args.transition_alpha if args.transitions != "none" else None,
        "dev_queries_with_center": int(dev_centers["hist_lat"].is_not_null().sum()),
    }
    del history
    dev_runs = retrieve(retrieval, "split", corpus, dev_queries, "dev", timings)

    t = time.perf_counter()
    items = ItemTable(corpus)
    embeddings = load_corpus_embeddings(retrieval.dense_config, "split", corpus, timings)
    item_vectors = torch.from_numpy(embeddings).to(default_device())
    del embeddings
    timings["item_table_s"] = time.perf_counter() - t
    dev_frame = add_history_features(
        pool_features(retrieval, dev_runs, dev_queries, "dev", items, item_vectors, timings),
        full_view(dev_queries, pairs),
        items,
        args.geo_history,
        args.transitions,
        args.transition_alpha,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model is not None:
        model_meta = json.loads(args.model.with_suffix(".json").read_text())
        if model_meta["features"] != features:
            raise SystemExit(f"{args.model} was trained with different features")
        model = CatBoostRanker()
        model.load_model(str(args.model))
        model_path, train_info = args.model, None
    else:
        model, train_info = train_model(
            args,
            retrieval,
            ranker,
            features,
            corpus,
            train_contexts,
            pairs,
            items,
            item_vectors,
            timings,
        )
        model_path = out_dir / "model.cbm"
        model.save_model(str(model_path))
        model_meta = {
            "features": features,
            "retrieval": dataclasses.asdict(retrieval),
            "ranker": dataclasses.asdict(ranker),
            "best_iteration": model.get_best_iteration(),
            "train_queries": args.train_queries,
            "history": history_info,
        }
        model_path.with_suffix(".json").write_text(
            json.dumps(model_meta, indent=2, ensure_ascii=False)
        )
    del train_contexts, pairs, item_vectors

    t = time.perf_counter()
    scores = model.predict(make_pool(dev_frame, features))
    timings["dev_predict_s"] = time.perf_counter() - t
    selected_rows = select_top(dev_frame, scores, dev_queries.height)
    selected = rows_to_ids(item_ids, selected_rows)

    positives = positive_outcomes(dev_queries, dev_frame, scores, items, item_ids, seen_items)
    per_query = query_outcomes(positives)
    summary = recall_report(dev_queries, selected, seen_items)
    if not np.isclose(per_query["recall"].mean(), summary["recall"]["@50"]):
        raise SystemExit("per-query recall does not match recall report")
    pool_sizes = dev_frame.group_by("q").len()["len"].to_numpy()
    no_center = per_query.filter(~pl.col("has_center"))

    per_query.write_parquet(out_dir / "per_query.parquet")
    positives.write_parquet(out_dir / "positives.parquet")
    report = {
        "experiment": name,
        "protocol": PROTOCOL,
        "parent": args.parent,
        "git": git_state(),
        "features": features,
        "dropped_features": args.drop_features,
        "history": history_info,
        "retrieval": dataclasses.asdict(retrieval),
        "ranker": dataclasses.asdict(ranker) if train_info else None,
        "model": {"path": str(model_path), **model_meta},
        "train": train_info,
        "recall@50": summary["recall"]["@50"],
        "pool_recall": float(per_query["pool_recall"].mean()),
        "pool_size": {
            "mean": float(pool_sizes.mean()),
            "p95": float(np.percentile(pool_sizes, 95)),
            "empty_queries": dev_queries.height - len(pool_sizes),
        },
        "vs_b0": compare(per_query, B0) if name != B0 else None,
        "vs_parent": compare(per_query, args.parent) if name != args.parent else None,
        "slices": {
            **summary["slices"],
            "no_center": {
                "n": no_center.height,
                "recall@50": float(no_center["recall"].mean()) if no_center.height else None,
            },
        },
        "error_map": error_map(positives),
        "feature_importance": dict(
            sorted(
                zip(
                    features,
                    map(float, model.get_feature_importance(type="PredictionValuesChange")),
                    strict=True,
                ),
                key=lambda kv: -kv[1],
            )
        ),
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (out_dir / "report.json").write_text(text)
    (out_dir / "error_examples.json").write_text(
        json.dumps(
            error_examples(positives, per_query, dev_queries, selected_rows, corpus, args.examples),
            indent=2,
            ensure_ascii=False,
        )
    )
    print(text)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
