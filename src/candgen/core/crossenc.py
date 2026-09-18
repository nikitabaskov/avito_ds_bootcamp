from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from sentence_transformers import CrossEncoder

from candgen.core import dense

CROSS_ENCODER_MODES = ("none", "bge_m3")
CROSS_ENCODER_FEATURES = {
    "none": [],
    "bge_m3": ["ce_score", "ce_rank", "ce_gap"],
}


@dataclass(frozen=True)
class CrossEncoderConfig:
    model: str = "BAAI/bge-reranker-v2-m3"
    revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    max_length: int = 256
    params_chars: int = 300
    text_chars: int = 1500
    top_k: int = 200
    batch_size: int = 64

    def tag(self) -> str:
        return (
            f"{self.model.split('/')[-1]}_len{self.max_length}"
            f"_p{self.params_chars}_c{self.text_chars}_k{self.top_k}"
        )


def scored_pairs(frame: pl.DataFrame, top_k: int) -> pl.DataFrame:
    return frame.filter(pl.col("rrf_rank") <= top_k).select("q", "row").sort("q", "row")


def pair_texts(
    config: CrossEncoderConfig, pairs: pl.DataFrame, queries: pl.DataFrame, corpus: pl.DataFrame
) -> tuple[list[str], list[str]]:
    query_text = [t.removeprefix("query: ") for t in dense.query_texts(queries, with_filters=True)]
    rows = pairs["row"].unique().sort()
    passages = [
        t.removeprefix("passage: ")[: config.text_chars]
        for t in dense.passage_texts(corpus[rows.to_numpy()], config.params_chars)
    ]
    passage_of = dict(zip(rows.to_list(), passages, strict=True))
    return (
        [query_text[q] for q in pairs["q"].to_list()],
        [passage_of[r] for r in pairs["row"].to_list()],
    )


def load_model(config: CrossEncoderConfig, device: str | None = None) -> CrossEncoder:
    device = device or dense.default_device()
    model = CrossEncoder(config.model, revision=config.revision, device=device)
    model.max_seq_length = config.max_length
    if device == "cuda":
        model.half()
    return model


def score(
    config: CrossEncoderConfig, pairs: pl.DataFrame, queries: pl.DataFrame, corpus: pl.DataFrame
) -> np.ndarray:
    query_text, passages = pair_texts(config, pairs, queries, corpus)
    order = np.argsort([-len(p) for p in passages], kind="stable")
    model = load_model(config)
    with torch.inference_mode():
        sorted_scores = model.predict(
            [(query_text[i], passages[i]) for i in order],
            batch_size=config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    scores = np.empty(len(order), dtype=np.float32)
    scores[order] = np.asarray(sorted_scores, dtype=np.float32)
    return scores


def add_cross_encoder_features(frame: pl.DataFrame, scored: pl.DataFrame) -> pl.DataFrame:
    ranked = scored.with_columns(
        ce_rank=pl.col("ce_score").rank("ordinal", descending=True).over("q").cast(pl.Float32),
        ce_gap=(pl.col("ce_score") - pl.col("ce_score").max().over("q")).cast(pl.Float32),
    )
    return (
        frame.join(ranked, on=["q", "row"], how="left")
        .select(*frame.columns, *CROSS_ENCODER_FEATURES["bge_m3"])
        .sort("q", "rrf_rank")
    )
