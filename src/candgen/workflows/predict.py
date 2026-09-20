"""Генерация answer.csv из итоговой модели с проверкой входов и результата."""

import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path

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
from candgen.core.dense import DenseConfig, config_for, default_device
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
from candgen.core.signals import CENTROID_FEATURES, ENCODER_FEATURES, FILTER_FEATURES
from candgen.core.submission import (
    answer_frame,
    read_answer,
    sha256_file,
    validate_answer,
)
from candgen.workflows.common import peak_rss_gb
from candgen.workflows.runs import (
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

DEFAULT_CONFIG = Path("configs/final.json")
QUERIES_PATH = INPUT_DIR / "benchmark_queries.parquet"
ITEMS_PATH = INPUT_DIR / "benchmark_items.parquet"


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(
                git("status", "--porcelain", "--", "src", "configs", "pyproject.toml", "uv.lock")
            ),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        # A downloaded source archive has no .git directory.
        return {"commit": None, "dirty": None}


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


GEO_KEYS = ("geo_km", "geo_k", "geo_weight", "geo_delta")


def model_config(meta: dict) -> RetrievalConfig:
    saved = meta["retrieval"]
    saved_dense = saved.get("dense_config", {})
    query_filters = saved_dense.get("query_filters", True)
    config = RetrievalConfig(
        **{
            k: saved[k]
            for k in ("global_k", "local_k", "radius_km", "radius_k", "region_k", *GEO_KEYS)
            if k in saved
        },
        dense_config=config_for(
            saved_dense.get("model", DenseConfig.model), query_filters=query_filters
        ),
    )
    current = json.loads(json.dumps(dataclasses.asdict(config)))
    if {k: current[k] for k in saved} != saved or set(current) - set(saved) - {
        "radius_km",
        "radius_k",
        "region_k",
        *GEO_KEYS,
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
        "encoder": (meta.get("signals") or {}).get("encoder"),
    }


def expected_features(spec: dict, config: RetrievalConfig) -> list[str]:
    return [
        *config.features(),
        *FIELD_FEATURES[spec["fields"]],
        *history_features(spec["geo"], spec["transitions"], spec["damping"]),
        *MICROCAT_FEATURES[spec["microcats"]],
        *(CENTROID_FEATURES if spec["centroid"] else []),
        *(FILTER_FEATURES if spec["filters"] else []),
        *(ENCODER_FEATURES if spec["encoder"] else []),
    ]


def needs_history(spec: dict, config: RetrievalConfig) -> bool:
    return (
        spec["geo"] != "none"
        or spec["transitions"] != "none"
        or bool(config.radius_k)
        or bool(config.geo_k)
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
    centers = (
        query_centers(views, items)
        if views is not None and (config.radius_k or config.geo_k)
        else None
    )
    runs = retrieve(config, corpus_name, corpus, queries, run_key, timings, centers)
    embeddings = load_corpus_embeddings(config.dense_config, corpus_name, corpus, timings)
    item_vectors = torch.from_numpy(embeddings).to(default_device())
    index = (
        microcat_index(config.dense_config, pairs, timings, exact_prior)
        if spec["microcats"] != "none"
        else None
    )
    runs |= region_runs(
        config, index, views, queries, corpus, items, item_vectors, run_key, timings
    )
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
    if index is not None:
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
        spec["encoder"],
        corpus_name,
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


def verify_file(path: Path, expected: str) -> None:
    """Reject changed inputs before reusing retrieval and embedding caches."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. See README.md for data and model preparation.")
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def main() -> None:
    """Run the submitted configuration and verify the resulting CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model", type=Path, help="Use a retrained model; exact submission hash is not enforced"
    )
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose another --output")
    meta = json.loads(args.config.read_text())
    submission = meta.pop("submission")
    model_path = args.model or Path(submission["model_path"])
    for name, digest in submission["inputs_sha256"].items():
        verify_file(INPUT_DIR / name, digest)
    if args.model is None:
        verify_file(model_path, submission["model_sha256"])
    else:
        # A retrained model must declare the same ordered feature contract.
        trained = json.loads(model_path.with_suffix(".json").read_text())
        for key in ("features", "retrieval", "ranker", "field_scores", "microcats", "signals"):
            if trained[key] != meta[key]:
                parser.error(f"retrained model has a different {key}")
    config = model_config(meta)
    queries = prepare_queries(pl.read_parquet(QUERIES_PATH))
    corpus = load_corpus("benchmark")
    query_ids = queries["query_id"].to_list()
    item_ids = corpus["item_id"].to_list()
    timings: dict[str, float] = {}
    started = time.perf_counter()
    predictions, model_meta = rank_with_model(model_path, meta, config, corpus, queries, timings)
    answer = answer_frame(query_ids, predictions)
    errors = validate_answer(answer, query_ids, item_ids)
    if errors:
        raise SystemExit("invalid answer:\n" + "\n".join(errors[:20]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answer.write_csv(args.output)
    written = read_answer(args.output)
    if not written.equals(answer) or validate_answer(written, query_ids, item_ids):
        raise SystemExit(f"{args.output} does not round-trip")
    digest = sha256_file(args.output)
    matches = digest == submission["answer_sha256"]
    timings["total_s"] = time.perf_counter() - started
    report = {
        "config": str(args.config),
        "model": model_meta,
        "git": git_state(),
        "answer_sha256": digest,
        "matches_submission": matches,
        "queries": len(query_ids),
        "corpus_items": corpus.height,
        "timings": timings,
        "peak_rss_gb": peak_rss_gb(),
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(
        json.dumps(
            {"output": str(args.output), "sha256": digest, "matches_submission": matches}, indent=2
        )
    )
    if args.model is None and not matches:
        raise SystemExit(
            "CSV differs from submission. Check artifact versions and execution environment."
        )


if __name__ == "__main__":
    main()
