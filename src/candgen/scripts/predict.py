import argparse
import dataclasses
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from candgen.core.data import (
    INPUT_DIR,
    build_contexts,
    build_corpus,
    load_corpus,
    load_train,
    prepare_queries,
)
from candgen.core.dense import DenseConfig, default_device
from candgen.core.features import ItemTable
from candgen.core.fields import FIELD_FEATURES, FieldScorer
from candgen.core.history import (
    add_history_features,
    full_view,
    history_features,
    history_pairs,
    query_centers,
)
from candgen.core.microcats import MICROCAT_FEATURES
from candgen.core.signals import CENTROID_FEATURES, FILTER_FEATURES
from candgen.core.submission import (
    ANSWER_K,
    answer_frame,
    read_answer,
    sha256_file,
    validate_answer,
)
from candgen.scripts.common import EXPERIMENTS_DIR, peak_rss_gb
from candgen.scripts.runs import (
    RetrievalConfig,
    fuse,
    load_corpus_embeddings,
    microcat_features,
    microcat_index,
    pool_features,
    retrieve,
    rows_to_ids,
    signal_features,
)

OUTPUT_DIR = Path("data/output")
QUERIES_PATH = INPUT_DIR / "benchmark_queries.parquet"
ITEMS_PATH = INPUT_DIR / "benchmark_items.parquet"


def next_output_dir(tag: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[:3]) for p in OUTPUT_DIR.iterdir() if re.match(r"\d{3}_", p.name)]
    path = (
        OUTPUT_DIR
        / f"{max(taken, default=0) + 1:03d}_{datetime.now().astimezone():%Y%m%d-%H%M%S}_{tag}"
    )
    path.mkdir()
    return path


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock")),
    }


def dev_summary(report_name: str, model_path: Path | None = None) -> dict | None:
    path = (
        model_path.parent / "report.json" if model_path else EXPERIMENTS_DIR / f"{report_name}.json"
    )
    if not path.exists():
        path = EXPERIMENTS_DIR / f"{report_name}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    recall = report["recall"]["@50"] if "recall" in report else report["recall@50"]
    return {
        "report": str(path),
        "recall@50": recall,
        "pool_recall": report.get("pool_recall", report.get("recall", {}).get("@1000")),
    }


def benchmark_history(timings: dict) -> tuple[pl.DataFrame, dict]:
    t = time.perf_counter()
    train = load_train()
    contexts = build_contexts(train)
    pairs = history_pairs(contexts, build_corpus(train))
    timings["history_s"] = time.perf_counter() - t
    return pairs, {
        "source": "train.parquet, all parts",
        "texts": contexts["query_text"].n_unique(),
        "contexts": contexts.height,
        "pairs_with_coords": pairs.height,
    }


def model_config(meta: dict) -> RetrievalConfig:
    saved = meta["retrieval"]
    query_filters = saved.get("dense_config", {}).get("query_filters", True)
    config = RetrievalConfig(
        **{k: saved[k] for k in ("global_k", "local_k", "radius_km", "radius_k") if k in saved},
        dense_config=DenseConfig(query_filters=query_filters),
    )
    current = json.loads(json.dumps(dataclasses.asdict(config)))
    if {k: current[k] for k in saved} != saved or set(current) - set(saved) - {
        "radius_km",
        "radius_k",
    }:
        raise SystemExit("model was trained with a different retrieval config")
    return config


def feature_spec(meta: dict) -> dict:
    history = meta.get("history") or {}
    return {
        "geo": history.get("geo_history", "none"),
        "transitions": history.get("transitions", "none"),
        "alpha": history.get("transition_alpha") or 0.0,
        "damping": history.get("geo_damping", False),
        "fields": meta.get("field_scores", "none"),
        "microcats": meta.get("microcats", "none"),
        "centroid": (meta.get("signals") or {}).get("centroid", False),
        "filters": (meta.get("signals") or {}).get("filters", False),
    }


def expected_features(spec: dict, config: RetrievalConfig) -> list[str]:
    return [
        *config.features(),
        *FIELD_FEATURES[spec["fields"]],
        *history_features(spec["geo"], spec["transitions"], spec["damping"]),
        *MICROCAT_FEATURES[spec["microcats"]],
        *(CENTROID_FEATURES if spec["centroid"] else []),
        *(FILTER_FEATURES if spec["filters"] else []),
    ]


def needs_history(spec: dict, config: RetrievalConfig) -> bool:
    return (
        spec["geo"] != "none"
        or spec["transitions"] != "none"
        or bool(config.radius_k)
        or spec["microcats"] != "none"
    )


def feature_frame(
    spec: dict,
    config: RetrievalConfig,
    corpus_name: str,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    run_key: str,
    items: ItemTable,
    pairs: pl.DataFrame | None,
    timings: dict,
    exact_prior: bool = False,
) -> pl.DataFrame:
    views = full_view(queries, pairs) if pairs is not None else None
    centers = query_centers(views, items) if views is not None and config.radius_k else None
    runs = retrieve(config, corpus_name, corpus, queries, run_key, timings, centers)
    embeddings = load_corpus_embeddings(config.dense_config, corpus_name, corpus, timings)
    item_vectors = torch.from_numpy(embeddings).to(default_device())
    frame = pool_features(config, runs, queries, run_key, items, item_vectors, timings)
    t = time.perf_counter()
    frame = FieldScorer(corpus, spec["fields"]).add(frame, queries)
    timings["field_scores_s"] = time.perf_counter() - t
    if views is not None:
        t = time.perf_counter()
        frame = add_history_features(
            frame,
            views,
            items,
            spec["geo"],
            spec["transitions"],
            spec["alpha"],
            spec["damping"],
        )
        timings["history_features_s"] = time.perf_counter() - t
    index = None
    if spec["microcats"] != "none":
        index = microcat_index(config.dense_config, pairs, timings, exact_prior)
        frame = microcat_features(
            config.dense_config, index, frame, views, queries, corpus, run_key, timings
        )
    return signal_features(
        config.dense_config,
        index,
        frame,
        views,
        queries,
        corpus,
        item_vectors,
        run_key,
        timings,
        spec["centroid"],
        spec["filters"],
    )


def rank_with_model(
    model_path: Path,
    meta: dict,
    config: RetrievalConfig,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    timings: dict,
    microcat_exact: bool = False,
) -> tuple[list[list[str]], dict]:
    from catboost import CatBoostRanker

    from candgen.core.ranker import make_pool, select_top

    spec = feature_spec(meta)
    if meta["features"] != expected_features(spec, config):
        raise SystemExit(f"{model_path} was trained with a different feature set")
    exact_prior = spec["microcats"] == "exact" or microcat_exact
    if microcat_exact and spec["microcats"] != "neighbors":
        raise SystemExit("--microcat-exact applies only to models with neighbor microcats")
    model = CatBoostRanker()
    model.load_model(str(model_path))
    t = time.perf_counter()
    items = ItemTable(corpus)
    timings["item_table_s"] = time.perf_counter() - t
    pairs, history_info = None, None
    if needs_history(spec, config):
        pairs, history_info = benchmark_history(timings)
        history_info |= {
            "geo_history": spec["geo"],
            "transitions": spec["transitions"],
            "transition_alpha": spec["alpha"] if spec["transitions"] != "none" else None,
            "geo_damping": spec["damping"],
            "microcats": spec["microcats"],
            "microcat_exact_prior": exact_prior,
        }
    frame = feature_frame(
        spec, config, "benchmark", corpus, queries, "benchmark", items, pairs, timings, exact_prior
    )
    del pairs
    t = time.perf_counter()
    scores = model.predict(make_pool(frame, meta["features"]))
    timings["predict_s"] = time.perf_counter() - t
    rows = select_top(frame, scores, queries.height)
    return rows_to_ids(corpus["item_id"].to_list(), rows), {
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        **meta,
        "inference_history": history_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["rrf", "catboost"], default="rrf")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--microcat-exact", action="store_true")
    args = parser.parse_args()
    if args.method == "catboost" and args.model is None:
        parser.error("--model is required for catboost")
    config = RetrievalConfig()
    meta: dict = {}
    if args.method == "catboost":
        meta = json.loads(args.model.with_suffix(".json").read_text())
        config = model_config(meta)

    queries = prepare_queries(pl.read_parquet(QUERIES_PATH))
    corpus = load_corpus("benchmark")
    item_ids = corpus["item_id"].to_list()
    query_ids = queries["query_id"].to_list()

    timings: dict[str, float] = {}
    started = time.perf_counter()
    if args.method == "rrf":
        runs = retrieve(config, "benchmark", corpus, queries, "benchmark", timings)
        predictions = rows_to_ids(item_ids, fuse(config, runs, timings))
        report_name = config.report_name("dev")
        model_meta = None
    else:
        predictions, model_meta = rank_with_model(
            args.model, meta, config, corpus, queries, timings, args.microcat_exact
        )
        report_name = f"catboost_dev_{args.model.stem.removeprefix('ranker_')}"
    timings["total_s"] = time.perf_counter() - started

    answer = answer_frame(query_ids, predictions)
    errors = validate_answer(answer, query_ids, item_ids)
    if errors:
        raise SystemExit("invalid answer:\n" + "\n".join(errors[:20]))

    out_dir = next_output_dir(args.tag or args.method)
    path = out_dir / "answer.csv"
    answer.write_csv(path)
    written = read_answer(path)
    if not written.equals(answer) or validate_answer(written, query_ids, item_ids):
        raise SystemExit(f"{path} does not round-trip")

    sizes = np.array([len(p[:ANSWER_K]) for p in predictions])
    meta = {
        "method": args.method,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "retrieval": dataclasses.asdict(config),
        "model": model_meta,
        "dev": dev_summary(report_name, args.model),
        "git": git_state(),
        "answer_sha256": sha256_file(path),
        "inputs_sha256": {p.name: sha256_file(p) for p in (QUERIES_PATH, ITEMS_PATH)},
        "queries": len(query_ids),
        "corpus_items": corpus.height,
        "answer_sizes": {"min": int(sizes.min()), "mean": float(sizes.mean())},
        "timings": timings,
        "peak_rss_gb": peak_rss_gb(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
