import functools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from candgen.core import bm25, dense
from candgen.core.data import ARTIFACTS_DIR
from candgen.core.retrieval import Hits, rrf_fuse

RUNS_DIR = ARTIFACTS_DIR / "runs"
GLOBAL_DEPTH = 1000
LOCAL_DEPTH = 500
CHANNELS = ("bm25", "dense")


@dataclass(frozen=True)
class RetrievalConfig:
    channels: tuple[str, ...] = CHANNELS
    local: bool = True
    global_k: int = 300
    local_k: int = 200
    rrf_k: int = 60
    bm25_config: bm25.BM25Config = field(
        default_factory=lambda: bm25.BM25Config(title_repeat=3, query_filters=True)
    )
    dense_config: dense.DenseConfig = field(default_factory=dense.DenseConfig)

    def list_names(self) -> list[str]:
        names = [f"{c}_global" for c in self.channels]
        if self.local:
            names += [f"{c}_local" for c in self.channels]
        return names

    def list_depth(self, name: str) -> int:
        return self.global_k if name.endswith("_global") else self.local_k

    def report_name(self, part: str) -> str:
        name = [f"rrf_{part}", "+".join(self.channels), f"g{self.global_k}"]
        if self.local:
            name.append(f"l{self.local_k}")
        name.append(f"k{self.rrf_k}")
        if "bm25" in self.channels:
            name.append(f"bm25-{self.bm25_config.tag()}")
        if "dense" in self.channels:
            filters = "_qf" if self.dense_config.query_filters else ""
            name.append(f"dense-p{self.dense_config.params_chars}{filters}")
        return "_".join(name)


def cached_run(
    path: Path, query_ids: list[str], compute: Callable[[], Hits], timings: dict
) -> Hits:
    if path.exists():
        data = np.load(path)
        if data["query_ids"].tolist() == query_ids:
            return data["rows"], data["scores"]
    t = time.perf_counter()
    rows, scores = compute()
    timings[f"{path.stem}_s"] = time.perf_counter() - t
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=rows, scores=scores, query_ids=np.array(query_ids))
    return rows, scores


def bm25_runs(
    config: bm25.BM25Config,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    run_dir: Path,
    timings: dict,
) -> dict[str, Hits]:
    @functools.cache
    def retriever() -> bm25.BM25Retriever:
        t = time.perf_counter()
        model = bm25.BM25Retriever(config)
        model.index(bm25.item_documents(corpus, config.title_repeat))
        timings["bm25_index_s"] = time.perf_counter() - t
        return model

    texts = bm25.query_texts(queries, config.query_filters)
    locations = queries["search_location_id"].to_numpy(), corpus["item_location_id"].to_numpy()
    base = f"bm25_{config.tag()}"
    query_ids = queries["query_id"].to_list()
    return {
        "bm25_global": cached_run(
            run_dir / f"{base}_global.npz",
            query_ids,
            lambda: retriever().search(texts, GLOBAL_DEPTH),
            timings,
        ),
        "bm25_local": cached_run(
            run_dir / f"{base}_local{LOCAL_DEPTH}.npz",
            query_ids,
            lambda: retriever().search_local(texts, *locations, LOCAL_DEPTH),
            timings,
        ),
    }


def query_vectors(
    config: dense.DenseConfig, queries: pl.DataFrame, run_dir: Path, timings: dict
) -> np.ndarray:
    filters = "_qf" if config.query_filters else ""
    path = run_dir / f"dense_{config.passage_tag()}{filters}_queries.npz"
    query_ids = queries["query_id"].to_list()
    if path.exists():
        data = np.load(path)
        if data["query_ids"].tolist() == query_ids:
            return data["vectors"]
    t = time.perf_counter()
    model = dense.load_model(config)
    timings["dense_model_load_s"] = time.perf_counter() - t
    t = time.perf_counter()
    vectors = dense.encode(
        model, dense.query_texts(queries, config.query_filters), config.batch_size
    )
    timings["dense_query_encode_s"] = time.perf_counter() - t
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, query_ids=np.array(query_ids))
    return vectors


def load_corpus_embeddings(
    config: dense.DenseConfig, corpus_name: str, corpus: pl.DataFrame, timings: dict
) -> np.ndarray:
    t = time.perf_counter()
    embeddings = dense.load_embeddings(
        dense.embedding_dir(corpus_name, config), corpus["item_id"].to_list()
    )
    timings["dense_embeddings_load_s"] = time.perf_counter() - t
    return embeddings


def dense_runs(
    config: dense.DenseConfig,
    corpus_name: str,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    run_dir: Path,
    timings: dict,
) -> dict[str, Hits]:
    @functools.cache
    def encoded() -> tuple[dense.DenseIndex, np.ndarray]:
        embeddings = load_corpus_embeddings(config, corpus_name, corpus, timings)
        return dense.DenseIndex(embeddings), query_vectors(config, queries, run_dir, timings)

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
        "dense_global": cached_run(run_dir / f"{base}_global.npz", query_ids, search, timings),
        "dense_local": cached_run(
            run_dir / f"{base}_local{LOCAL_DEPTH}.npz", query_ids, search_local, timings
        ),
    }


def retrieve(
    config: RetrievalConfig,
    corpus_name: str,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    run_key: str,
    timings: dict,
) -> dict[str, Hits]:
    run_dir = RUNS_DIR / run_key
    runs: dict[str, Hits] = {}
    if "bm25" in config.channels:
        runs |= bm25_runs(config.bm25_config, corpus, queries, run_dir, timings)
    if "dense" in config.channels:
        runs |= dense_runs(config.dense_config, corpus_name, corpus, queries, run_dir, timings)
    return runs


def fuse(config: RetrievalConfig, runs: dict[str, Hits], timings: dict) -> list[np.ndarray]:
    t = time.perf_counter()
    fused = rrf_fuse(
        [runs[name][0][:, : config.list_depth(name)] for name in config.list_names()], config.rrf_k
    )
    timings["rrf_s"] = time.perf_counter() - t
    return fused


def rows_to_ids(item_ids: list[str], rows: Sequence[np.ndarray]) -> list[list[str]]:
    return [[item_ids[r] for r in row.tolist() if r >= 0] for row in rows]
