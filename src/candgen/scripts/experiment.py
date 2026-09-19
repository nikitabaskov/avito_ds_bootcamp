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
from candgen.core.dense import DenseConfig, default_device
from candgen.core.diagnostics import error_map, positive_outcomes, query_outcomes
from candgen.core.evaluation import compare_per_query, recall_report
from candgen.core.features import ItemTable, attach_labels
from candgen.core.fields import FIELD_FEATURES, FIELD_MODES, FieldScorer
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
    query_centers,
)
from candgen.core.metrics import recall_at_k
from candgen.core.microcats import MICROCAT_FEATURES, MICROCAT_MODES, MicrocatIndex
from candgen.core.ranker import RankerConfig, full_recall_curve, make_pool, select_top, train_ranker
from candgen.core.signals import CENTROID_FEATURES, FILTER_FEATURES
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.predict import git_state
from candgen.scripts.runs import (
    LOCAL_DEPTH,
    RetrievalConfig,
    load_corpus_embeddings,
    microcat_features,
    microcat_index,
    pool_features,
    region_runs,
    retrieve,
    rows_to_ids,
    signal_features,
)

PROTOCOL = "v2"
TREES = 500
B0 = "EXP-000/b0"
VALID_BASE_QUERIES = 6000
VALID_SHARE = 0.1
TREE_POINTS = (10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000)
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


def train_frames(
    args: argparse.Namespace,
    retrieval: RetrievalConfig,
    corpus: pl.DataFrame,
    train_contexts: pl.DataFrame,
    pairs: pl.DataFrame,
    items: ItemTable,
    item_vectors: torch.Tensor,
    scorer: FieldScorer,
    index: MicrocatIndex | None,
    timings: dict,
) -> dict:
    train_queries = sample_eval_queries(train_contexts, args.train_queries, SEED)
    valid_texts = fixed_valid_texts(train_contexts)
    train_key = f"train{args.train_queries}"
    roles = train_queries.select(
        q=pl.int_range(0, pl.len(), dtype=pl.Int32),
        valid=pl.col("query_text").is_in(valid_texts.implode()),
        excluded=valid_bucket(train_queries["query_text"])
        & ~pl.col("query_text").is_in(valid_texts.implode()),
    )
    views = crossfit_views(train_queries, pairs, roles.filter("valid").select("q"))
    runs = retrieve(
        retrieval,
        "split",
        corpus,
        train_queries,
        train_key,
        timings,
        query_centers(views, items) if retrieval.radius_k or retrieval.geo_k else None,
    )
    runs |= region_runs(
        retrieval, index, views, train_queries, corpus, items, item_vectors, train_key, timings
    )
    frame = pool_features(retrieval, runs, train_queries, train_key, items, item_vectors, timings)
    t = time.perf_counter()
    frame = scorer.add(frame, train_queries)
    timings["train_field_scores_s"] = time.perf_counter() - t
    t = time.perf_counter()
    frame = add_history_features(
        frame,
        views,
        items,
        args.geo_history,
        args.transitions,
        args.transition_alpha,
        args.geo_damping,
    )
    frame = microcat_features(
        retrieval.dense_config, index, frame, views, train_queries, corpus, train_key, timings
    )
    frame = signal_features(
        retrieval.dense_config,
        index,
        frame,
        views,
        train_queries,
        corpus,
        item_vectors,
        train_key,
        timings,
        args.neighbor_centroid,
        args.filter_match,
    )
    frame = attach_labels(
        frame,
        train_queries["item_ids"].to_list(),
        corpus["item_id"].to_list(),
    )
    timings["train_history_s"] = time.perf_counter() - t
    with_positive = frame.group_by("q").agg(pl.col("label").max() > 0).filter("label").select("q")
    frame = frame.join(roles, on="q").filter(~pl.col("excluded")).sort("q", "rrf_rank")
    full_valid_frame = frame.filter("valid")
    full_valid_queries = train_queries.with_row_index("q").join(
        roles.filter("valid").select("q"), on="q", how="semi"
    )
    frame = frame.join(with_positive, on="q", how="semi").sort("q", "rrf_rank")
    return {
        "fit": frame.filter(~pl.col("valid")),
        "valid": frame.filter(pl.col("valid")),
        "full_valid": full_valid_frame,
        "full_valid_queries": full_valid_queries,
        "stats": {
            "sampled_queries": train_queries.height,
            "queries_with_positive_in_pool": with_positive.height,
            "excluded_valid_bucket_queries": int(roles["excluded"].sum()),
        },
    }


def train_model(
    ranker: RankerConfig,
    features: list[str],
    frames: dict,
    corpus: pl.DataFrame,
    timings: dict,
) -> tuple[CatBoostRanker, dict]:
    fit_frame, valid_frame = frames["fit"], frames["valid"]
    full_valid_frame, full_valid_queries = frames["full_valid"], frames["full_valid_queries"]
    t = time.perf_counter()
    model = train_ranker(fit_frame, valid_frame, ranker, features)
    timings["train_s"] = time.perf_counter() - t
    t = time.perf_counter()
    points = [n for n in TREE_POINTS if n < model.tree_count_] + [model.tree_count_]
    full_valid = full_recall_curve(
        model, full_valid_frame, full_valid_queries, corpus["item_id"].to_list(), features, points
    )
    timings["full_valid_s"] = time.perf_counter() - t
    return model, {
        **frames["stats"],
        "fit_queries": fit_frame["q"].n_unique(),
        "valid_queries": valid_frame["q"].n_unique(),
        "valid_texts_file": str(VALID_TEXTS_PATH),
        "fit_rows": fit_frame.height,
        "positive_rows": int(fit_frame["label"].sum()),
        "best_iteration": model.get_best_iteration(),
        "trees": model.tree_count_,
        "best_valid": model.get_best_score().get("validation"),
        "validation": {
            "queries": full_valid_queries.height,
            "queries_with_positive_in_pool": valid_frame["q"].n_unique(),
            "full_recall@50": full_valid,
            "selection": (
                "conditional_early_stopping" if ranker.early_stopping_rounds else "fixed_iterations"
            ),
        },
    }


def compare(per_query: pl.DataFrame, reference: str) -> dict | None:
    path = EXPERIMENTS_DIR / reference / "per_query.parquet"
    if not path.exists():
        return None
    try:
        return {"reference": reference, **compare_per_query(per_query, pl.read_parquet(path))}
    except ValueError as error:
        raise SystemExit(f"{reference}: {error}") from error


def tree_curve(
    model: CatBoostRanker,
    frame: pl.DataFrame,
    features: list[str],
    queries: pl.DataFrame,
    item_ids: list[str],
) -> dict:
    trees = model.tree_count_
    points = [n for n in TREE_POINTS if n < trees] + [trees]
    pool = make_pool(frame, features)
    relevant = queries["item_ids"].to_list()
    dev = {}
    for n in points:
        rows = select_top(frame, model.predict(pool, ntree_end=n), queries.height)
        dev[n] = float(recall_at_k(rows_to_ids(item_ids, rows), relevant, 50).mean())
    valid = model.get_evals_result().get("validation", {}).get("RecallAt:top=50", [])
    return {
        "dev_recall@50": dev,
        "valid_recall@50": {n: float(valid[n - 1]) for n in points if n <= len(valid)},
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


def build_parser(required: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=required)
    parser.add_argument("--variant", required=required)
    parser.add_argument("--parent", default=B0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--drop-features", nargs="*", default=[])
    parser.add_argument("--geo-history", choices=GEO_MODES, default="none")
    parser.add_argument("--transitions", choices=TRANSITION_MODES, default="none")
    parser.add_argument("--transition-alpha", type=float, default=10.0)
    parser.add_argument("--geo-damping", action="store_true")
    parser.add_argument("--field-scores", choices=FIELD_MODES, default="none")
    parser.add_argument("--microcats", choices=MICROCAT_MODES, default="none")
    parser.add_argument("--neighbor-centroid", action="store_true")
    parser.add_argument("--filter-match", action="store_true")
    parser.add_argument("--global-k", type=int, default=RetrievalConfig.global_k)
    parser.add_argument("--local-k", type=int, default=RetrievalConfig.local_k)
    parser.add_argument("--radius-km", type=float, default=RetrievalConfig.radius_km)
    parser.add_argument("--radius-k", type=int, default=RetrievalConfig.radius_k)
    parser.add_argument("--region-k", type=int, default=RetrievalConfig.region_k)
    parser.add_argument("--geo-km", type=float, default=RetrievalConfig.geo_km)
    parser.add_argument("--geo-k", type=int, default=RetrievalConfig.geo_k)
    parser.add_argument("--geo-weight", type=float, default=RetrievalConfig.geo_weight)
    parser.add_argument("--geo-delta", type=float, default=RetrievalConfig.geo_delta)
    parser.add_argument("--e5-query-no-filters", action="store_true")
    parser.add_argument("--train-queries", type=int, default=VALID_BASE_QUERIES)
    parser.add_argument("--iterations", type=int, default=TREES)
    parser.add_argument("--loss-function", default=RankerConfig.loss_function)
    parser.add_argument("--learning-rate", type=float, default=RankerConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=RankerConfig.depth)
    parser.add_argument("--seed", type=int, default=RankerConfig.random_seed)
    parser.add_argument("--early-stopping-rounds", type=int, default=0)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default=RankerConfig.task_type)
    parser.add_argument("--examples", type=int, default=30)
    return parser


def configure(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[RetrievalConfig, RankerConfig, list[str]]:
    if not 0 <= args.local_k <= LOCAL_DEPTH:
        parser.error(f"--local-k must be between 0 and {LOCAL_DEPTH} (cached search depth)")
    retrieval = RetrievalConfig(
        global_k=args.global_k,
        local_k=args.local_k,
        radius_km=args.radius_km,
        radius_k=args.radius_k,
        region_k=args.region_k,
        geo_km=args.geo_km,
        geo_k=args.geo_k,
        geo_weight=args.geo_weight,
        geo_delta=args.geo_delta,
        dense_config=DenseConfig(query_filters=not args.e5_query_no_filters),
    )
    if retrieval.radius_k and retrieval.radius_km <= 0:
        parser.error("--radius-k needs a positive --radius-km")
    if retrieval.geo_k and retrieval.geo_km <= 0:
        parser.error("--geo-k needs a positive --geo-km")
    if args.geo_damping and args.geo_history == "none":
        parser.error("--geo-damping needs --geo-history")
    if (args.neighbor_centroid or args.region_k) and args.microcats == "none":
        parser.error("--neighbor-centroid and --region-k need --microcats")
    available = [
        *retrieval.features(),
        *FIELD_FEATURES[args.field_scores],
        *history_features(args.geo_history, args.transitions, args.geo_damping),
        *MICROCAT_FEATURES[args.microcats],
        *(CENTROID_FEATURES if args.neighbor_centroid else []),
        *(FILTER_FEATURES if args.filter_match else []),
    ]
    unknown = set(args.drop_features) - set(available)
    if unknown:
        parser.error(f"unknown features: {sorted(unknown)}")
    features = [f for f in available if f not in args.drop_features]
    ranker = RankerConfig(
        loss_function=args.loss_function,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.seed,
        task_type=args.task_type,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    return retrieval, ranker, features


def prepare(args: argparse.Namespace, retrieval: RetrievalConfig, timings: dict) -> dict:
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
        "geo_damping": args.geo_damping,
        "dev_queries_with_center": int(dev_centers["hist_lat"].is_not_null().sum()),
    }
    del history

    t = time.perf_counter()
    items = ItemTable(corpus)
    dev_views = full_view(dev_queries, pairs)
    dev_runs = retrieve(
        retrieval,
        "split",
        corpus,
        dev_queries,
        "dev",
        timings,
        query_centers(dev_views, items) if retrieval.radius_k or retrieval.geo_k else None,
    )
    embeddings = load_corpus_embeddings(retrieval.dense_config, "split", corpus, timings)
    item_vectors = torch.from_numpy(embeddings).to(default_device())
    del embeddings
    timings["item_table_s"] = time.perf_counter() - t
    t = time.perf_counter()
    scorer = FieldScorer(corpus, args.field_scores)
    timings["field_index_s"] = time.perf_counter() - t
    index = (
        microcat_index(retrieval.dense_config, pairs, timings, args.microcats == "exact")
        if args.microcats != "none"
        else None
    )
    dev_runs |= region_runs(
        retrieval, index, dev_views, dev_queries, corpus, items, item_vectors, "dev", timings
    )
    dev_frame = add_history_features(
        scorer.add(
            pool_features(retrieval, dev_runs, dev_queries, "dev", items, item_vectors, timings),
            dev_queries,
        ),
        dev_views,
        items,
        args.geo_history,
        args.transitions,
        args.transition_alpha,
        args.geo_damping,
    )
    dev_frame = microcat_features(
        retrieval.dense_config, index, dev_frame, dev_views, dev_queries, corpus, "dev", timings
    )
    dev_frame = signal_features(
        retrieval.dense_config,
        index,
        dev_frame,
        dev_views,
        dev_queries,
        corpus,
        item_vectors,
        "dev",
        timings,
        args.neighbor_centroid,
        args.filter_match,
    )
    signal_info = {"centroid": args.neighbor_centroid, "filters": args.filter_match}
    return {
        "corpus": corpus,
        "dev_queries": dev_queries,
        "seen_items": seen_items,
        "item_ids": item_ids,
        "train_contexts": train_contexts,
        "pairs": pairs,
        "history_info": history_info,
        "items": items,
        "item_vectors": item_vectors,
        "scorer": scorer,
        "index": index,
        "dev_frame": dev_frame,
        "signal_info": signal_info,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    name = f"{args.exp}/{args.variant}"
    out_dir = EXPERIMENTS_DIR / name
    if (out_dir / "report.json").exists():
        raise SystemExit(f"{out_dir} already has a report")
    retrieval, ranker, features = configure(parser, args)

    started = time.perf_counter()
    git = git_state()
    timings: dict[str, float] = {}
    setup = prepare(args, retrieval, timings)
    corpus, dev_queries, seen_items = setup["corpus"], setup["dev_queries"], setup["seen_items"]
    item_ids, train_contexts, pairs = setup["item_ids"], setup["train_contexts"], setup["pairs"]
    history_info, items, item_vectors = setup["history_info"], setup["items"], setup["item_vectors"]
    scorer, index, dev_frame = setup["scorer"], setup["index"], setup["dev_frame"]
    signal_info = setup["signal_info"]
    del setup

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model is not None:
        model_meta = json.loads(args.model.with_suffix(".json").read_text())
        if model_meta["features"] != features:
            raise SystemExit(f"{args.model} was trained with different features")
        model = CatBoostRanker()
        model.load_model(str(args.model))
        model_path, train_info = args.model, None
    else:
        frames = train_frames(
            args,
            retrieval,
            corpus,
            train_contexts,
            pairs,
            items,
            item_vectors,
            scorer,
            index,
            timings,
        )
        model, train_info = train_model(ranker, features, frames, corpus, timings)
        del frames
        model_path = out_dir / "model.cbm"
        model.save_model(str(model_path))
        model_meta = {
            "features": features,
            "retrieval": dataclasses.asdict(retrieval),
            "ranker": dataclasses.asdict(ranker),
            "best_iteration": model.get_best_iteration(),
            "train_queries": args.train_queries,
            "history": history_info,
            "field_scores": args.field_scores,
            "microcats": args.microcats,
            "signals": signal_info,
        }
        model_path.with_suffix(".json").write_text(
            json.dumps(model_meta, indent=2, ensure_ascii=False)
        )
    del train_contexts, pairs, item_vectors, index

    t = time.perf_counter()
    scores = model.predict(make_pool(dev_frame, features))
    timings["dev_predict_s"] = time.perf_counter() - t
    selected_rows = select_top(dev_frame, scores, dev_queries.height)
    selected = rows_to_ids(item_ids, selected_rows)
    curve = tree_curve(model, dev_frame, features, dev_queries, item_ids) if train_info else None

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
        "git": git,
        "features": features,
        "dropped_features": args.drop_features,
        "history": history_info,
        "field_scores": args.field_scores,
        "microcats": args.microcats,
        "signals": signal_info,
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
        "tree_curve": curve,
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
