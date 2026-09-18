import os
from dataclasses import dataclass

import bm25s
import numpy as np
import polars as pl
import Stemmer
from bm25s.tokenization import Tokenizer

from candgen.core.data import normalize_text
from candgen.core.retrieval import Groups, Hits, order_hits, search_groups, search_local


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    title_repeat: int = 1
    query_filters: bool = False

    def tag(self) -> str:
        filters = "_qf" if self.query_filters else ""
        return f"t{self.title_repeat}{filters}_k{self.k1}_b{self.b}"


def item_documents(items: pl.DataFrame, title_repeat: int = 1) -> list[str]:
    title = normalize_text(pl.col("item_title_raw"))
    parts = [title] * title_repeat + [
        normalize_text(pl.col("item_infm_params_text")),
        normalize_text(pl.col("item_description_raw")),
    ]
    return items.select(pl.concat_str(parts, separator=" ")).to_series().to_list()


def query_texts(queries: pl.DataFrame, with_filters: bool = False) -> list[str]:
    parts = [pl.col("query_text")]
    if with_filters:
        parts.append(normalize_text(pl.col("search_infm_params_text")))
    return queries.select(pl.concat_str(parts, separator=" ")).to_series().to_list()


class BM25Retriever:
    def __init__(self, config: BM25Config):
        self.config = config
        self.tokenizer = Tokenizer(stemmer=Stemmer.Stemmer("russian"), stopwords="ru")
        self.model = bm25s.BM25(k1=config.k1, b=config.b, method=config.method)

    def index(self, documents: list[str]) -> None:
        tokens = self.tokenizer.tokenize(documents, update_vocab=True, return_as="ids")
        self.model.index(tokens, show_progress=True)
        self.size = len(documents)

    def search(self, queries: list[str], k: int) -> Hits:
        return self._retrieve(self._tokenize(queries), k)

    def search_local(
        self,
        queries: list[str],
        query_locations: np.ndarray,
        item_locations: np.ndarray,
        k: int,
    ) -> Hits:
        tokens = self._tokenize(queries)

        def search(query_rows: np.ndarray, item_rows: np.ndarray, depth: int) -> Hits:
            mask = np.zeros(len(item_locations), dtype=np.float32)
            mask[item_rows] = 1.0
            return self._retrieve([tokens[i] for i in query_rows], depth, mask)

        return search_local(search, query_locations, item_locations, k)

    def search_groups(self, queries: list[str], groups: Groups, k: int) -> Hits:
        tokens = self._tokenize(queries)

        def search(query_rows: np.ndarray, item_rows: np.ndarray, depth: int) -> Hits:
            mask = np.zeros(self.size, dtype=np.float32)
            mask[item_rows] = 1.0
            return self._retrieve([tokens[i] for i in query_rows], depth, mask)

        return search_groups(search, groups, len(queries), k)

    def pair_scores(self, queries: list[str], q: np.ndarray, rows: np.ndarray) -> np.ndarray:
        empty = self.tokenizer.get_vocab_dict().get("")
        tokens = [[t for t in ids if t != empty] for ids in self._tokenize(queries)]
        out = np.zeros(len(q), dtype=np.float32)
        bounds = np.flatnonzero(np.diff(q)) + 1
        starts = np.concatenate([[0], bounds]).astype(np.int64)
        ends = np.concatenate([bounds, [len(q)]]).astype(np.int64)
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            if start < end and tokens[q[start]]:
                scores = self.model.get_scores_from_ids(tokens[q[start]])
                out[start:end] = scores[rows[start:end]]
        return out

    def _tokenize(self, queries: list[str]) -> list[list[int]]:
        return self.tokenizer.tokenize(queries, update_vocab=False, return_as="ids")

    def _retrieve(
        self, tokens: list[list[int]], k: int, weight_mask: np.ndarray | None = None
    ) -> Hits:
        rows, scores = self.model.retrieve(
            tokens,
            k=k,
            n_threads=min(os.cpu_count() or 1, len(tokens)),
            weight_mask=weight_mask,
            show_progress=weight_mask is None,
        )
        rows, scores = order_hits(rows.astype(np.int64), scores)
        rows[scores <= 0] = -1
        return rows, scores
