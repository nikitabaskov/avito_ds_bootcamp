import argparse
import dataclasses
import functools
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch

from candgen.core import bm25, dense
from candgen.core.data import ARTIFACTS_DIR
from candgen.core.evaluation import POOL_KS, recall_report
from candgen.core.retrieval import Hits, rrf_fuse
from candgen.scripts.common import load_eval, peak_rss_gb, write_report

RUNS_DIR = ARTIFACTS_DIR / "runs"
GLOBAL_DEPTH = max(POOL_KS)
LOCAL_DEPTH = 500
CHANNELS = ("bm25", "dense")


def cached_run(
    path: Path, query_ids: list[str], compute: Callable[[], Hits], timings: dict
) -> np.ndarray:
    if path.exists():
        data = np.load(path)
        if data["query_ids"].tolist() == query_ids:
            return data["rows"]
    t = time.perf_counter()
    rows, scores = compute()
    timings[f"{path.stem}_s"] = time.perf_counter() - t
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=rows, scores=scores, query_ids=np.array(query_ids))
    return rows


def bm25_runs(
    config: bm25.BM25Config,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    part: str,
    timings: dict,
) -> dict[str, np.ndarray]:
    @functools.cache
    def retriever() -> bm25.BM25Retriever:
        t = time.perf_counter()
        model = bm25.BM25Retriever(config)
        model.index(bm25.item_documents(corpus, config.title_repeat))
        timings["bm25_index_s"] = time.perf_counter() - t
        return model

    texts = bm25.query_texts(queries, config.query_filters)
    locations = queries["search_location_id"].to_numpy(), corpus["item_location_id"].to_numpy()
    base = RUNS_DIR / part / f"bm25_{config.tag()}"
    query_ids = queries["query_id"].to_list()
    return {
        "bm25_global": cached_run(
            base.with_name(f"{base.name}_global.npz"),
            query_ids,
            lambda: retriever().search(texts, GLOBAL_DEPTH),
            timings,
        ),
        "bm25_local": cached_run(
            base.with_name(f"{base.name}_local{LOCAL_DEPTH}.npz"),
            query_ids,
            lambda: retriever().search_local(texts, *locations, LOCAL_DEPTH),
            timings,
        ),
    }


def dense_runs(
    config: dense.DenseConfig,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    part: str,
    timings: dict,
) -> dict[str, np.ndarray]:
    @functools.cache
    def encoded() -> tuple[dense.DenseIndex, np.ndarray]:
        t = time.perf_counter()
        embeddings = dense.load_embeddings(
            dense.embedding_dir("split", config), corpus["item_id"].to_list()
        )
        index = dense.DenseIndex(embeddings)
        timings["dense_index_load_s"] = time.perf_counter() - t
        t = time.perf_counter()
        model = dense.load_model(config)
        timings["dense_model_load_s"] = time.perf_counter() - t
        t = time.perf_counter()
        vectors = dense.encode(
            model, dense.query_texts(queries, config.query_filters), config.batch_size
        )
        timings["dense_query_encode_s"] = time.perf_counter() - t
        return index, vectors

    def search() -> Hits:
        index, vectors = encoded()
        return index.search(vectors, GLOBAL_DEPTH)

    def search_local() -> Hits:
        index, vectors = encoded()
        return index.search_local(
            vectors,
            queries["search_location_id"].to_numpy(),
            corpus["item_location_id"].to_numpy(),
            LOCAL_DEPTH,
        )

    filters = "_qf" if config.query_filters else ""
    base = f"dense_{config.passage_tag()}{filters}"
    query_ids = queries["query_id"].to_list()
    return {
        "dense_global": cached_run(
            RUNS_DIR / part / f"{base}_global.npz", query_ids, search, timings
        ),
        "dense_local": cached_run(
            RUNS_DIR / part / f"{base}_local{LOCAL_DEPTH}.npz", query_ids, search_local, timings
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["dev", "test"], default="dev")
    parser.add_argument("--channels", nargs="+", choices=CHANNELS, default=list(CHANNELS))
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-k", type=int, default=300)
    parser.add_argument("--local-k", type=int, default=200)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--title-repeat", type=int, default=3)
    parser.add_argument("--bm25-query-filters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dense-query-filters", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--params-chars", type=int, default=dense.DenseConfig.params_chars)
    args = parser.parse_args()
    if args.global_k > GLOBAL_DEPTH or args.local_k > LOCAL_DEPTH:
        parser.error(f"global-k <= {GLOBAL_DEPTH}, local-k <= {LOCAL_DEPTH}")
    channels = [c for c in CHANNELS if c in args.channels]
    bm25_config = bm25.BM25Config(
        title_repeat=args.title_repeat, query_filters=args.bm25_query_filters
    )
    dense_config = dense.DenseConfig(
        params_chars=args.params_chars, query_filters=args.dense_query_filters
    )

    corpus, queries, seen_items = load_eval(args.part)
    item_ids = corpus["item_id"].to_list()

    timings: dict[str, float] = {}
    runs: dict[str, np.ndarray] = {}
    if "bm25" in channels:
        runs |= bm25_runs(bm25_config, corpus, queries, args.part, timings)
    if "dense" in channels:
        runs |= dense_runs(dense_config, corpus, queries, args.part, timings)

    def to_ids(rows: np.ndarray) -> list[str]:
        return [item_ids[r] for r in rows.tolist() if r >= 0]

    channel_reports = {}
    for name, rows in runs.items():
        full = recall_report(queries, [to_ids(r) for r in rows], seen_items)
        channel_reports[name] = {
            "depth": int(rows.shape[1]),
            **{k: full[k] for k in ("recall", "mean_pool_size", "empty_pool_queries")},
        }

    fused_inputs = [runs[f"{c}_global"][:, : args.global_k] for c in channels]
    if args.local:
        fused_inputs += [runs[f"{c}_local"][:, : args.local_k] for c in channels]
    t = time.perf_counter()
    fused = rrf_fuse(fused_inputs, args.rrf_k)
    timings["rrf_s"] = time.perf_counter() - t

    report = {
        "method": "rrf",
        "part": args.part,
        "config": {
            "channels": channels,
            "local": args.local,
            "global_k": args.global_k,
            "local_k": args.local_k,
            "rrf_k": args.rrf_k,
            "bm25": dataclasses.asdict(bm25_config) if "bm25" in channels else None,
            "dense": dataclasses.asdict(dense_config) if "dense" in channels else None,
        },
        "corpus_items": corpus.height,
        **recall_report(queries, [to_ids(r) for r in fused], seen_items),
        "channels": channel_reports,
        "timings": timings,
        "peak_rss_gb": peak_rss_gb(),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else None,
    }
    name = [f"rrf_{args.part}", "+".join(channels), f"g{args.global_k}"]
    if args.local:
        name.append(f"l{args.local_k}")
    name.append(f"k{args.rrf_k}")
    if "bm25" in channels:
        name.append(f"bm25-{bm25_config.tag()}")
    if "dense" in channels:
        name.append(
            f"dense-p{dense_config.params_chars}{'_qf' if dense_config.query_filters else ''}"
        )
    write_report("_".join(name), report)


if __name__ == "__main__":
    main()
