import re
from collections.abc import Sequence

import numpy as np
import polars as pl
import Stemmer
import torch

from candgen.core.data import normalize_text
from candgen.core.retrieval import Hits

TOKEN_RE = re.compile(r"\w+")
PARAMS_CHARS = 500
EARTH_RADIUS_KM = 6371.0

LIST_FEATURES = [
    f"{kind}_{name}"
    for name in ("bm25_global", "bm25_local", "dense_global", "dense_local")
    for kind in ("rank", "score")
]
FEATURES = [
    *LIST_FEATURES,
    "rrf",
    "rrf_rank",
    "n_lists",
    "dense_sim",
    "title_overlap",
    "params_overlap",
    "loc_match",
    "dist_km",
    "cat_match",
    "query_cat_zero",
    "filters_empty",
    "query_words",
    "location_items",
    "price",
    "rating",
    "reviews",
    "phone_hidden",
    "message_forbidden",
    "title_chars",
    "params_chars",
    "description_chars",
]


def candidate_pool(runs: dict[str, Hits], depths: dict[str, int], rrf_k: int) -> pl.DataFrame:
    frames = []
    for name, depth in depths.items():
        rows, scores = runs[name]
        rows, scores = rows[:, :depth], scores[:, :depth]
        q, pos = np.nonzero(rows >= 0)
        frames.append(
            pl.DataFrame(
                {
                    "q": q.astype(np.int32),
                    "row": rows[q, pos].astype(np.int64),
                    "list": name,
                    "rank": (pos + 1).astype(np.float32),
                    "score": scores[q, pos].astype(np.float32),
                }
            )
        )
    pool = pl.concat(frames).pivot(
        on="list", index=["q", "row"], values=["rank", "score"], aggregate_function="first"
    )
    pool = pool.with_columns(
        pl.lit(None, dtype=pl.Float32).alias(c) for c in LIST_FEATURES if c not in pool.columns
    )
    ranks = [pl.col(f"rank_{name}") for name in depths]
    return (
        pool.with_columns(
            rrf=pl.sum_horizontal(
                [(1.0 / (rrf_k + r.cast(pl.Float64))).fill_null(0.0) for r in ranks]
            ),
            n_lists=pl.sum_horizontal([r.is_not_null().cast(pl.Int32) for r in ranks]),
        )
        .sort(["q", "rrf", "row"], descending=[False, True, False])
        .with_columns(rrf_rank=pl.int_range(1, pl.len() + 1).over("q").cast(pl.Float32))
    )


class StemSets:
    def __init__(self) -> None:
        self.stemmer = Stemmer.Stemmer("russian")

    def __call__(self, texts: Sequence[str]) -> list[frozenset[str]]:
        tokens = [TOKEN_RE.findall(t) for t in texts]
        vocab = sorted({w for words in tokens for w in words})
        stems = dict(zip(vocab, self.stemmer.stemWords(vocab), strict=True))
        return [frozenset(stems[w] for w in words) for words in tokens]


def overlap(
    query_sets: list[frozenset[str]], item_sets: list[frozenset[str]], q, rows
) -> np.ndarray:
    out = np.zeros(len(q), dtype=np.float32)
    for i, (qi, ri) in enumerate(zip(q.tolist(), rows.tolist(), strict=True)):
        terms = query_sets[qi]
        if terms:
            out[i] = len(terms & item_sets[ri]) / len(terms)
    return out


class ItemTable:
    def __init__(self, corpus: pl.DataFrame):
        stems = StemSets()
        self.title_stems = stems(
            corpus.select(normalize_text(pl.col("item_title_raw"))).to_series().to_list()
        )
        self.params_stems = stems(
            corpus.select(
                normalize_text(pl.col("item_infm_params_text")).str.slice(0, PARAMS_CHARS)
            )
            .to_series()
            .to_list()
        )
        self.frame = corpus.select(
            row=pl.int_range(0, pl.len(), dtype=pl.Int64),
            item_location_id=pl.col("item_location_id"),
            item_category_id=pl.col("item_category_id"),
            lat=pl.col("item_latitude").cast(pl.Float64),
            lon=pl.col("item_longitude").cast(pl.Float64),
            price=pl.col("item_price").cast(pl.Float32),
            rating=pl.col("item_rating").cast(pl.Float32),
            reviews=pl.col("item_rating_reviews_count").cast(pl.Float32),
            phone_hidden=pl.col("item_is_phone_hidden").cast(pl.Float32),
            message_forbidden=pl.col("item_is_message_forbidden").cast(pl.Float32),
            title_chars=pl.col("item_title_raw").str.len_chars().fill_null(0).cast(pl.Float32),
            params_chars=pl.col("item_infm_params_text")
            .str.len_chars()
            .fill_null(0)
            .cast(pl.Float32),
            description_chars=pl.col("item_description_raw")
            .str.len_chars()
            .fill_null(0)
            .cast(pl.Float32),
        )
        self.locations = self.frame.group_by("item_location_id").agg(
            loc_lat=pl.col("lat").median(),
            loc_lon=pl.col("lon").median(),
            location_items=pl.len().cast(pl.Float32),
        )


def pair_similarity(
    item_vectors: torch.Tensor, query_vectors: np.ndarray, q: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    out = np.empty(len(q), dtype=np.float32)
    bounds = np.flatnonzero(np.diff(q)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(q)]])
    device = item_vectors.device
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if start == end:
            continue
        items = item_vectors[torch.tensor(rows[start:end], device=device)]
        query = torch.from_numpy(query_vectors[q[start]]).to(device, item_vectors.dtype)
        out[start:end] = (items @ query).float().cpu().numpy()
    return out


def haversine_km(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    phi1, phi2 = lat1.radians(), lat2.radians()
    a = ((phi2 - phi1) / 2).sin() ** 2 + phi1.cos() * phi2.cos() * (
        ((lon2.radians() - lon1.radians()) / 2).sin() ** 2
    )
    return (2 * EARTH_RADIUS_KM * a.sqrt().arcsin()).cast(pl.Float32)


def build_features(
    pool: pl.DataFrame,
    queries: pl.DataFrame,
    items: ItemTable,
    query_vectors: np.ndarray,
    item_vectors: torch.Tensor,
) -> pl.DataFrame:
    query_frame = queries.select(
        q=pl.int_range(0, pl.len(), dtype=pl.Int32),
        search_location_id=pl.col("search_location_id"),
        search_category=pl.col("search_category"),
        query_cat_zero=(pl.col("search_category") == 0).cast(pl.Float32),
        filters_empty=(pl.col("search_infm_params_text") == "").cast(pl.Float32),
        query_words=pl.col("query_text").str.split(" ").list.len().cast(pl.Float32),
    ).join(
        items.locations,
        left_on="search_location_id",
        right_on="item_location_id",
        how="left",
    )
    query_stems = StemSets()(queries["query_text"].to_list())

    frame = (
        pool.join(items.frame, on="row", how="left")
        .join(query_frame, on="q", how="left")
        .sort("q", "rrf_rank")
    )
    q, rows = frame["q"].to_numpy(), frame["row"].to_numpy()
    return frame.with_columns(
        dense_sim=pair_similarity(item_vectors, query_vectors, q, rows),
        title_overlap=overlap(query_stems, items.title_stems, q, rows),
        params_overlap=overlap(query_stems, items.params_stems, q, rows),
        loc_match=(pl.col("item_location_id") == pl.col("search_location_id")).cast(pl.Float32),
        cat_match=(pl.col("item_category_id") == pl.col("search_category")).cast(pl.Float32),
        dist_km=haversine_km(pl.col("lat"), pl.col("lon"), pl.col("loc_lat"), pl.col("loc_lon")),
        location_items=pl.col("location_items").fill_null(0.0),
    ).select("q", "row", *FEATURES)


def attach_labels(
    frame: pl.DataFrame, positives: Sequence[Sequence[str]], item_ids: Sequence[str]
) -> pl.DataFrame:
    row_of = {item: i for i, item in enumerate(item_ids)}
    pairs = [(q, row_of[i]) for q, rel in enumerate(positives) for i in rel if i in row_of]
    labels = pl.DataFrame(
        {"q": [q for q, _ in pairs], "row": [r for _, r in pairs]},
        schema={"q": pl.Int32, "row": pl.Int64},
    ).with_columns(label=pl.lit(1, dtype=pl.Int8))
    return (
        frame.join(labels, on=["q", "row"], how="left")
        .with_columns(pl.col("label").fill_null(0))
        .sort("q", "rrf_rank")
    )
