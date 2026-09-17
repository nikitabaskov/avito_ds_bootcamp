from collections.abc import Sequence

import numpy as np
import polars as pl

from candgen.core.features import ItemTable, StemSets

TOP_K = 50
RANK_BUCKETS = [50, 100, 300]


def model_ranks(frame: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    return (
        frame.select("q", "row", "rrf_rank")
        .with_columns(score=pl.Series(scores, dtype=pl.Float64))
        .sort(["q", "score", "rrf_rank"], descending=[False, True, False])
        .with_columns(model_rank=pl.int_range(1, pl.len() + 1).over("q").cast(pl.Int32))
        .drop("score")
    )


def positive_outcomes(
    queries: pl.DataFrame,
    frame: pl.DataFrame,
    scores: np.ndarray,
    items: ItemTable,
    item_ids: Sequence[str],
    seen_items: set[str],
) -> pl.DataFrame:
    row_of = {item: i for i, item in enumerate(item_ids)}
    relevant = queries["item_ids"].to_list()
    pairs = pl.DataFrame(
        {
            "q": [q for q, rel in enumerate(relevant) for _ in rel],
            "item_id": [i for rel in relevant for i in rel],
        },
        schema={"q": pl.Int32, "item_id": pl.String},
    ).with_columns(
        row=pl.col("item_id").replace_strict(row_of, default=None, return_dtype=pl.Int64),
        n_pos=pl.len().over("q").cast(pl.Int32),
        item_seen=pl.col("item_id").is_in(list(seen_items)),
    )

    words = queries["query_text"].str.split(" ").list.len()
    filter_stems = StemSets()(queries["search_infm_params_text"].to_list())
    query_frame = queries.select(
        q=pl.int_range(0, pl.len(), dtype=pl.Int32),
        query_id="query_id",
        search_location_id="search_location_id",
        filters_set=pl.col("search_infm_params_text") != "",
        words=pl.when(words >= 4).then(pl.lit("4+")).otherwise(words.cast(pl.String)),
        has_center=pl.col("search_location_id").is_in(
            items.locations["item_location_id"].implode()
        ),
    )
    ranks = model_ranks(frame, scores).join(
        frame.select("q", "row", "n_lists", "dist_km", "loc_match"), on=["q", "row"]
    )
    out = (
        pairs.join(query_frame, on="q", how="left")
        .join(items.frame.select("row", "item_location_id"), on="row", how="left")
        .join(ranks, on=["q", "row"], how="left")
        .with_columns(
            same_location=pl.col("item_location_id") == pl.col("search_location_id"),
            in_pool=pl.col("model_rank").is_not_null(),
            in_top50=pl.col("model_rank") <= TOP_K,
        )
        .with_columns(pl.col("in_top50").fill_null(False))
        .sort("q", "item_id")
    )
    q, rows = out["q"].to_list(), out["row"].to_list()
    filter_overlap = np.full(len(q), np.nan, dtype=np.float32)
    for i, (qi, ri) in enumerate(zip(q, rows, strict=True)):
        terms = filter_stems[qi]
        if terms and ri is not None:
            filter_overlap[i] = len(terms & items.params_stems[ri]) / len(terms)
    return out.with_columns(filter_overlap=pl.Series(filter_overlap).fill_nan(None))


def query_outcomes(positives: pl.DataFrame) -> pl.DataFrame:
    return (
        positives.group_by("q", maintain_order=True)
        .agg(
            "query_id",
            "words",
            "filters_set",
            "has_center",
            n_pos=pl.len(),
            recall=pl.col("in_top50").mean(),
            pool_recall=pl.col("in_pool").mean(),
            any_other_location=(~pl.col("same_location").fill_null(False)).any(),
        )
        .with_columns(pl.col("query_id", "words", "filters_set", "has_center").list.first())
        .sort("q")
    )


def rank_bucket(rank: pl.Expr) -> pl.Expr:
    expr = pl.when(rank.is_null()).then(pl.lit("not_in_pool"))
    lower = 0
    for upper in RANK_BUCKETS:
        expr = expr.when(rank <= upper).then(pl.lit(f"{lower + 1}-{upper}"))
        lower = upper
    return expr.otherwise(pl.lit(f">{lower}"))


def loss_breakdown(positives: pl.DataFrame, by: str | pl.Expr, n_queries: int) -> list[dict]:
    weight = 1.0 / (pl.col("n_pos") * n_queries)
    key = by if isinstance(by, pl.Expr) else pl.col(by)
    return (
        positives.group_by(key.alias("group"))
        .agg(
            positives=pl.len(),
            queries=pl.col("q").n_unique(),
            recall=pl.col("in_top50").mean(),
            pool_recall=pl.col("in_pool").mean(),
            retrieval_loss=(weight * (~pl.col("in_pool"))).sum(),
            selection_loss=(weight * (pl.col("in_pool") & ~pl.col("in_top50"))).sum(),
        )
        .sort("group", nulls_last=True)
        .to_dicts()
    )


def error_map(positives: pl.DataFrame) -> dict:
    n_queries = positives["q"].n_unique()
    weight = 1.0 / (pl.col("n_pos") * n_queries)
    totals = positives.select(
        recall=(weight * pl.col("in_top50")).sum(),
        retrieval_loss=(weight * (~pl.col("in_pool"))).sum(),
        selection_loss=(weight * (pl.col("in_pool") & ~pl.col("in_top50"))).sum(),
    ).to_dicts()[0]
    lost = positives.filter(~pl.col("in_top50"))
    kept = positives.filter(pl.col("in_top50"))
    return {
        "queries": n_queries,
        "positives": positives.height,
        **totals,
        "queries_with_retrieval_miss": positives.filter(~pl.col("in_pool"))["q"].n_unique(),
        "queries_with_selection_miss": positives.filter(pl.col("in_pool") & ~pl.col("in_top50"))[
            "q"
        ].n_unique(),
        "by": {
            "same_location": loss_breakdown(positives, "same_location", n_queries),
            "has_center": loss_breakdown(positives, "has_center", n_queries),
            "words": loss_breakdown(positives, "words", n_queries),
            "filters_set": loss_breakdown(positives, "filters_set", n_queries),
            "item_seen": loss_breakdown(positives, "item_seen", n_queries),
            "n_lists": loss_breakdown(positives, "n_lists", n_queries),
            "model_rank": loss_breakdown(positives, rank_bucket(pl.col("model_rank")), n_queries),
            "multi_positive": loss_breakdown(positives, pl.col("n_pos") > 1, n_queries),
        },
        "filter_overlap_mean": {
            "top50_positives": kept["filter_overlap"].mean(),
            "lost_positives": lost["filter_overlap"].mean(),
        },
        "dist_km_median": {
            "top50_positives": kept["dist_km"].median(),
            "selection_lost_positives": lost.filter(pl.col("in_pool"))["dist_km"].median(),
        },
    }
