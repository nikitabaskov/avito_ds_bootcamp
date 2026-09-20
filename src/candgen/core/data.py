"""Подготовка контекстов и разбиение по тексту без пересечения запросов."""

import hashlib
from collections.abc import Iterable
from pathlib import Path

import polars as pl

INPUT_DIR = Path("data/input")
ARTIFACTS_DIR = Path("data/artifacts")
SPLIT_DIR = ARTIFACTS_DIR / "split"

SEED = 42
EVAL_QUERIES_PER_PART = 2452

CONTEXT_COLS = [
    "query_text",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]


def normalize_text(expr: pl.Expr) -> pl.Expr:
    return (
        expr.fill_null("")
        .str.to_lowercase()
        .str.replace_all("ё", "е", literal=True)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def stable_hash(values: Iterable[str], seed: int) -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"{seed}\x1f{v}".encode()).digest()[:8], "big")
        for v in values
    ]


def prepare_queries(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        query_text=normalize_text(pl.col("search_query")),
        search_infm_params_text=pl.col("search_infm_params_text").fill_null(""),
    )


def load_train(path: Path = INPUT_DIR / "train.parquet") -> pl.DataFrame:
    return prepare_queries(pl.read_parquet(path))


def assign_parts(texts: pl.Series, seed: int = SEED) -> pl.DataFrame:
    unique = texts.unique().sort()
    bucket = pl.col("bucket")
    part = (
        pl.when(bucket < 8_000)
        .then(pl.lit("train"))
        .when(bucket < 9_000)
        .then(pl.lit("dev"))
        .otherwise(pl.lit("test"))
    )
    buckets = pl.Series(stable_hash(unique, seed), dtype=pl.UInt64) % 10_000
    return pl.DataFrame({"query_text": unique, "bucket": buckets}).select("query_text", part=part)


def context_id(df: pl.DataFrame) -> pl.Series:
    keys = (
        df.select(
            pl.concat_str([pl.col(c).cast(pl.String) for c in CONTEXT_COLS], separator="\x1f")
        )
        .to_series()
        .to_list()
    )
    return pl.Series("query_id", [hashlib.sha256(k.encode()).hexdigest()[:16] for k in keys])


def build_contexts(pairs: pl.DataFrame) -> pl.DataFrame:
    contexts = pairs.group_by(CONTEXT_COLS).agg(item_ids=pl.col("item_id").unique().sort())
    return contexts.with_columns(context_id(contexts)).sort("query_id")


def sample_eval_queries(
    contexts: pl.DataFrame, n: int = EVAL_QUERIES_PER_PART, seed: int = SEED
) -> pl.DataFrame:
    texts = contexts["query_text"].unique()
    picked = (
        pl.DataFrame(
            {
                "query_text": texts,
                "order": pl.Series(
                    stable_hash((f"text\x1f{t}" for t in texts), seed), dtype=pl.UInt64
                ),
            }
        )
        .sort("order", "query_text")
        .head(n)
    )
    chosen = contexts.join(picked.select("query_text"), on="query_text", how="semi")
    return (
        chosen.with_columns(order=pl.Series(stable_hash(chosen["query_id"], seed), dtype=pl.UInt64))
        .sort("order", "query_id")
        .unique("query_text", keep="first", maintain_order=True)
        .drop("order")
        .sort("query_id")
    )


def build_corpus(train: pl.DataFrame) -> pl.DataFrame:
    return train.select(ITEM_COLS).unique("item_id", keep="first").sort("item_id")


def load_corpus(name: str) -> pl.DataFrame:
    if name == "split":
        return pl.read_parquet(SPLIT_DIR / "corpus.parquet")
    if name == "benchmark":
        return (
            pl.read_parquet(INPUT_DIR / "benchmark_items.parquet").select(ITEM_COLS).sort("item_id")
        )
    raise ValueError(f"unknown corpus: {name}")


def repeated_text_queries(
    contexts: pl.DataFrame, excluded_texts: pl.Series, n: int, seed: int = SEED
) -> pl.DataFrame:
    repeated = contexts.filter(~pl.col("query_text").is_in(excluded_texts.implode())).filter(
        pl.len().over("query_text") >= 2
    )
    return sample_eval_queries(repeated, n, seed)
