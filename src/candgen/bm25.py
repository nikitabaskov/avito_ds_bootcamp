import os
from dataclasses import dataclass

import bm25s
import numpy as np
import polars as pl
import Stemmer
from bm25s.tokenization import Tokenizer

from candgen.data import normalize_text


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    title_repeat: int = 1
    query_filters: bool = False


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

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.tokenizer.tokenize(queries, update_vocab=False, return_as="ids")
        rows, scores = self.model.retrieve(tokens, k=k, n_threads=os.cpu_count() or 1)
        order = np.lexsort((rows, -scores), axis=1)
        rows = np.take_along_axis(rows, order, axis=1)
        scores = np.take_along_axis(scores, order, axis=1)
        rows[scores <= 0] = -1
        return rows, scores
