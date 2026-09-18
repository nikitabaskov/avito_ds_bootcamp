import functools
import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch

from candgen.core import bm25, dense
from candgen.core.data import ARTIFACTS_DIR
from candgen.core.features import FEATURES, ItemTable, build_features, candidate_pool, list_features
from candgen.core.history import HistoryView
from candgen.core.microcats import MicrocatIndex, add_microcat_features
from candgen.core.retrieval import Groups, Hits, radius_groups, rrf_fuse

RUNS_DIR = ARTIFACTS_DIR / "runs"
GLOBAL_DEPTH = 1000
LOCAL_DEPTH = 500
RADIUS_DEPTH = 100
CHANNELS = ("bm25", "dense")


@dataclass(frozen=True)
class RetrievalConfig:
    channels: tuple[str, ...] = CHANNELS
    local: bool = True
    global_k: int = 300
    local_k: int = 200
    rrf_k: int = 60
    radius_km: float = 0.0
    radius_k: int = 0
    bm25_config: bm25.BM25Config = field(
        default_factory=lambda: bm25.BM25Config(title_repeat=3, query_filters=True)
    )
    dense_config: dense.DenseConfig = field(default_factory=dense.DenseConfig)

    def list_names(self) -> list[str]:
        names = [f"{c}_global" for c in self.channels]
        if self.local:
            names += [f"{c}_local" for c in self.channels]
        if self.radius_k:
            names += [f"{c}_radius" for c in self.channels]
        return names

    def list_depth(self, name: str) -> int:
        if name.endswith("_radius"):
            return self.radius_k
        return self.global_k if name.endswith("_global") else self.local_k

    def features(self) -> list[str]:
        return [*FEATURES, *list_features([n for n in self.list_names() if n.endswith("_radius")])]

    def report_name(self, part: str) -> str:
        name = [f"rrf_{part}", "+".join(self.channels), f"g{self.global_k}"]
        if self.local:
            name.append(f"l{self.local_k}")
        if self.radius_k:
            name.append(f"r{self.radius_km:g}x{self.radius_k}")
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
    radius: tuple[str, Groups] | None = None,
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
    runs = {
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
    if radius:
        tag, groups = radius
        runs["bm25_radius"] = cached_run(
            run_dir / f"{base}_{tag}.npz",
            query_ids,
            lambda: retriever().search_groups(texts, groups, RADIUS_DEPTH),
            timings,
        )
    return runs


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


def text_vectors(
    config: dense.DenseConfig, texts: list[str], path: Path, timings: dict
) -> np.ndarray:
    if path.exists():
        data = np.load(path)
        if data["texts"].tolist() == texts:
            return data["vectors"]
    t = time.perf_counter()
    model = dense.load_model(config)
    vectors = dense.encode(
        model, dense.query_texts(pl.DataFrame({"query_text": texts}), False), config.batch_size
    )
    timings[f"{path.stem}_encode_s"] = time.perf_counter() - t
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, texts=np.array(texts))
    return vectors


def microcat_index(config: dense.DenseConfig, pairs: pl.DataFrame, timings: dict) -> MicrocatIndex:
    texts = pairs["query_text"].unique().sort().to_list()
    digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]
    vectors = text_vectors(config, texts, RUNS_DIR / "history" / f"texts_{digest}.npz", timings)
    return MicrocatIndex(texts, vectors, dense.default_device())


def microcat_features(
    config: dense.DenseConfig,
    index: MicrocatIndex | None,
    frame: pl.DataFrame,
    views: list[HistoryView],
    queries: pl.DataFrame,
    corpus: pl.DataFrame,
    run_key: str,
    timings: dict,
) -> pl.DataFrame:
    if index is None:
        return frame
    vectors = text_vectors(
        config, queries["query_text"].to_list(), RUNS_DIR / run_key / "e5_texts.npz", timings
    )
    t = time.perf_counter()
    frame = add_microcat_features(frame, views, index, vectors, corpus["item_microcat_id"])
    timings[f"{run_key}_microcats_s"] = time.perf_counter() - t
    return frame


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
    radius: tuple[str, Groups] | None = None,
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
    runs = {
        "dense_global": cached_run(run_dir / f"{base}_global.npz", query_ids, search, timings),
        "dense_local": cached_run(
            run_dir / f"{base}_local{LOCAL_DEPTH}.npz", query_ids, search_local, timings
        ),
    }
    if radius:
        tag, groups = radius

        def search_radius() -> Hits:
            index, vectors = encoded()
            return index.search_groups(vectors, groups, RADIUS_DEPTH)

        runs["dense_radius"] = cached_run(
            run_dir / f"{base}_{tag}.npz", query_ids, search_radius, timings
        )
    return runs


def retrieve(
    config: RetrievalConfig,
    corpus_name: str,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    run_key: str,
    timings: dict,
    centers: np.ndarray | None = None,
) -> dict[str, Hits]:
    run_dir = RUNS_DIR / run_key
    radius = None
    if config.radius_k:
        if centers is None or len(centers) != queries.height:
            raise ValueError("radius lists need one center per query")
        if config.radius_k > RADIUS_DEPTH:
            raise ValueError(f"radius_k is capped by cached depth {RADIUS_DEPTH}")
        t = time.perf_counter()
        coords = corpus.select(
            pl.col("item_latitude", "item_longitude").cast(pl.Float64).fill_null(np.nan)
        ).to_numpy()
        digest = hashlib.sha256(np.ascontiguousarray(centers, dtype=np.float64).tobytes())
        radius = (
            f"radius{config.radius_km:g}_{RADIUS_DEPTH}_c{digest.hexdigest()[:12]}",
            radius_groups(centers, coords, config.radius_km),
        )
        timings["radius_groups_s"] = time.perf_counter() - t
    runs: dict[str, Hits] = {}
    if "bm25" in config.channels:
        runs |= bm25_runs(config.bm25_config, corpus, queries, run_dir, timings, radius)
    if "dense" in config.channels:
        runs |= dense_runs(
            config.dense_config, corpus_name, corpus, queries, run_dir, timings, radius
        )
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


def pool_features(
    config: RetrievalConfig,
    runs: dict[str, Hits],
    queries: pl.DataFrame,
    run_key: str,
    items: ItemTable,
    item_vectors: torch.Tensor,
    timings: dict,
) -> pl.DataFrame:
    t = time.perf_counter()
    pool = candidate_pool(
        runs, {name: config.list_depth(name) for name in config.list_names()}, config.rrf_k
    )
    vectors = query_vectors(config.dense_config, queries, RUNS_DIR / run_key, timings)
    frame = build_features(pool, queries, items, vectors, item_vectors)
    timings[f"{run_key}_features_s"] = time.perf_counter() - t
    return frame
