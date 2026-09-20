"""Полевые BM25-признаки; отключены в итоговой конфигурации."""

import polars as pl

from candgen.core.bm25 import BM25Config, BM25Retriever
from candgen.core.data import normalize_text

FIELD_MODES = ("none", "fields", "filters")
FIELDS = {
    "title": "item_title_raw",
    "params": "item_infm_params_text",
    "description": "item_description_raw",
}
FIELD_FEATURES = {
    "none": [],
    "fields": [f"bm25_{name}" for name in FIELDS],
    "filters": [*(f"bm25_{name}" for name in FIELDS), "bm25_filters"],
}


class FieldScorer:
    def __init__(self, corpus: pl.DataFrame, mode: str):
        self.mode = mode
        self.retrievers: dict[str, BM25Retriever] = {}
        if mode == "none":
            return
        for name, column in FIELDS.items():
            retriever = BM25Retriever(BM25Config())
            retriever.index(
                corpus.select(normalize_text(pl.col(column)).fill_null("")).to_series().to_list()
            )
            self.retrievers[name] = retriever

    def features(self) -> list[str]:
        return FIELD_FEATURES[self.mode]

    def add(self, frame: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
        if self.mode == "none":
            return frame
        frame = frame.sort("q", "rrf_rank")
        q, rows = frame["q"].to_numpy(), frame["row"].to_numpy()
        texts = queries["query_text"].to_list()
        columns = {
            f"bm25_{name}": retriever.pair_scores(texts, q, rows)
            for name, retriever in self.retrievers.items()
        }
        if self.mode == "filters":
            filters = (
                queries.select(normalize_text(pl.col("search_infm_params_text")).fill_null(""))
                .to_series()
                .to_list()
            )
            columns["bm25_filters"] = self.retrievers["params"].pair_scores(filters, q, rows)
        return frame.with_columns(**columns)
