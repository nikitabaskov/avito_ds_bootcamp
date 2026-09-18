import numpy as np
import polars as pl

from candgen.core.data import SEED, stable_hash
from candgen.core.features import ItemTable, haversine_km

FOLDS = 5
GEO_MODES = ("none", "fallback", "full")
TRANSITION_MODES = ("none", "prob", "full")
GEO_FEATURES = {
    "none": [],
    "fallback": ["center_source", "hist_pairs"],
    "full": ["center_source", "hist_pairs", "hist_dist_km", "hist_spread_km"],
}
TRANSITION_PROB = ["trans_prob", "trans_pairs", "trans_support", "trans_unknown"]
TRANSITION_FEATURES = {
    "none": [],
    "prob": TRANSITION_PROB,
    "full": [*TRANSITION_PROB, "trans_lift", "trans_targets", "trans_outside_share"],
}
DAMPING_FEATURES = ["geo_unreliable", "log_dist_clean"]
UNRELIABLE_SPREAD_KM = 75.0
UNRELIABLE_PAIRS = 5
DIST_CLIP_KM = 1000.0
CENTER_COLS = ["hist_lat", "hist_lon", "hist_pairs", "hist_items", "hist_spread_km"]
SOURCE_COLS = ["trans_support", "trans_targets", "trans_outside_share"]

type HistoryView = tuple[pl.DataFrame, pl.DataFrame]


def history_features(geo: str, transitions: str, damping: bool = False) -> list[str]:
    return [
        *GEO_FEATURES[geo],
        *TRANSITION_FEATURES[transitions],
        *(DAMPING_FEATURES if damping else []),
    ]


def text_fold(texts: pl.Series, folds: int = FOLDS) -> pl.Series:
    hashes = pl.Series(stable_hash((f"history\x1f{t}" for t in texts), SEED), dtype=pl.UInt64)
    return (hashes % folds).alias("fold")


def history_pairs(contexts: pl.DataFrame, corpus: pl.DataFrame) -> pl.DataFrame:
    item_frame = corpus.select(
        "item_id",
        "item_location_id",
        "item_microcat_id",
        lat=pl.col("item_latitude").cast(pl.Float64),
        lon=pl.col("item_longitude").cast(pl.Float64),
    )
    pairs = (
        contexts.select("query_text", "search_location_id", item_id="item_ids")
        .explode("item_id", empty_as_null=False)
        .drop_nulls("item_id")
        .join(item_frame, on="item_id")
    )
    return pairs.with_columns(text_fold(pairs["query_text"]))


def query_locations(queries: pl.DataFrame) -> pl.DataFrame:
    return queries.select(
        q=pl.int_range(0, pl.len(), dtype=pl.Int32), search_location_id="search_location_id"
    )


def full_view(queries: pl.DataFrame, pairs: pl.DataFrame) -> list[HistoryView]:
    return [(query_locations(queries), pairs)]


def crossfit_views(
    queries: pl.DataFrame, pairs: pl.DataFrame, holdout_q: pl.DataFrame, folds: int = FOLDS
) -> list[HistoryView]:
    located = query_locations(queries).with_columns(text_fold(queries["query_text"], folds))
    views = [
        (
            located.filter(pl.col("fold") == f).join(holdout_q, on="q", how="anti").drop("fold"),
            pairs.filter(pl.col("fold") != f),
        )
        for f in range(folds)
    ]
    views.append((located.join(holdout_q, on="q", how="semi").drop("fold"), pairs))
    return views


def location_centers(pairs: pl.DataFrame) -> pl.DataFrame:
    located = pairs.drop_nulls(["lat", "lon"])
    centers = located.group_by("search_location_id").agg(
        hist_lat=pl.col("lat").median(),
        hist_lon=pl.col("lon").median(),
        hist_pairs=pl.len().cast(pl.Float32),
        hist_items=pl.col("item_id").n_unique().cast(pl.Float32),
    )
    spread = (
        located.join(centers, on="search_location_id")
        .group_by("search_location_id")
        .agg(
            hist_spread_km=haversine_km(
                pl.col("lat"), pl.col("lon"), pl.col("hist_lat"), pl.col("hist_lon")
            ).median()
        )
    )
    return centers.join(spread, on="search_location_id").select("search_location_id", *CENTER_COLS)


def query_centers(views: list[HistoryView], items: ItemTable) -> np.ndarray:
    parts = [
        queries.join(
            items.locations.filter(pl.col("location_items") > 0),
            left_on="search_location_id",
            right_on="item_location_id",
            how="left",
        ).join(location_centers(pairs), on="search_location_id", how="left")
        for queries, pairs in views
    ]
    return (
        pl.concat(parts)
        .sort("q")
        .select(
            lat=pl.coalesce("loc_lat", "hist_lat").fill_null(np.nan),
            lon=pl.coalesce("loc_lon", "hist_lon").fill_null(np.nan),
        )
        .to_numpy()
    )


def corpus_location_share(items: ItemTable) -> pl.DataFrame:
    return items.locations.select(
        item_location_id="item_location_id",
        p_corpus=(pl.col("location_items") / pl.col("location_items").sum()).cast(pl.Float64),
    )


def location_transitions(
    pairs: pl.DataFrame, share: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    counts = (
        pairs.group_by("search_location_id", "item_location_id")
        .agg(trans_pairs=pl.len().cast(pl.Float64))
        .join(share, on="item_location_id", how="left")
    )
    in_corpus = pl.col("p_corpus").is_not_null()
    sources = counts.group_by("search_location_id").agg(
        trans_support=pl.col("trans_pairs").filter(in_corpus).sum(),
        trans_targets=in_corpus.sum().cast(pl.Float64),
        trans_outside_share=pl.col("trans_pairs").filter(~in_corpus).sum()
        / pl.col("trans_pairs").sum(),
    )
    return counts.filter(in_corpus).select(
        "search_location_id", "item_location_id", "trans_pairs"
    ), sources


def view_features(
    frame: pl.DataFrame,
    view: HistoryView,
    items: ItemTable,
    share: pl.DataFrame,
    geo: str,
    transitions: str,
    alpha: float,
    damping: bool = False,
) -> pl.DataFrame:
    queries, pairs = view
    joined = frame.join(queries, on="q").join(
        items.frame.select("row", "lat", "lon", "item_location_id"), on="row", how="left"
    )
    columns: dict[str, pl.Expr] = {}
    if geo != "none":
        joined = joined.join(location_centers(pairs), on="search_location_id", how="left")
        hist_dist = haversine_km(
            pl.col("lat"), pl.col("lon"), pl.col("hist_lat"), pl.col("hist_lon")
        )
        has_corpus_center = pl.col("location_items") > 0
        columns |= {
            "dist_km": pl.when(has_corpus_center).then(pl.col("dist_km")).otherwise(hist_dist),
            "center_source": pl.when(has_corpus_center)
            .then(0.0)
            .when(pl.col("hist_lat").is_not_null())
            .then(1.0)
            .otherwise(2.0)
            .cast(pl.Float32),
            "hist_pairs": pl.col("hist_pairs").fill_null(0.0),
        }
        if geo == "full":
            columns |= {"hist_dist_km": hist_dist, "hist_spread_km": pl.col("hist_spread_km")}
        if damping:
            joined = joined.with_columns(**columns)
            unreliable = (pl.col("center_source") == 1.0) & (
                (pl.col("hist_spread_km") > UNRELIABLE_SPREAD_KM)
                | (pl.col("hist_pairs") < UNRELIABLE_PAIRS)
            )
            columns = {
                "geo_unreliable": unreliable.cast(pl.Float32),
                "log_dist_clean": pl.when(unreliable)
                .then(-1.0)
                .otherwise(pl.col("dist_km").clip(0.0, DIST_CLIP_KM).log1p())
                .cast(pl.Float32),
            }
    if transitions != "none":
        counts, sources = location_transitions(pairs, share)
        joined = (
            joined.join(sources, on="search_location_id", how="left")
            .join(counts, on=["search_location_id", "item_location_id"], how="left")
            .join(share, on="item_location_id", how="left")
            .with_columns(
                pl.col("trans_pairs", "trans_support", "trans_targets").fill_null(0.0),
                pl.col("p_corpus").fill_null(0.0),
            )
        )
        prob = (pl.col("trans_pairs") + alpha * pl.col("p_corpus")) / (
            pl.col("trans_support") + alpha
        )
        columns |= {
            "trans_prob": prob.cast(pl.Float32),
            "trans_pairs": pl.col("trans_pairs").cast(pl.Float32),
            "trans_support": pl.col("trans_support").cast(pl.Float32),
            "trans_unknown": (pl.col("trans_pairs") == 0).cast(pl.Float32),
        }
        if transitions == "full":
            columns |= {
                "trans_lift": pl.when(pl.col("p_corpus") > 0)
                .then((1.0 + prob / pl.col("p_corpus")).log())
                .cast(pl.Float32),
                "trans_targets": pl.col("trans_targets").cast(pl.Float32),
                "trans_outside_share": pl.col("trans_outside_share").cast(pl.Float32),
            }
    return joined.with_columns(**columns).select(
        *frame.columns, *history_features(geo, transitions, damping)
    )


def add_history_features(
    frame: pl.DataFrame,
    views: list[HistoryView],
    items: ItemTable,
    geo: str,
    transitions: str,
    alpha: float,
    damping: bool = False,
) -> pl.DataFrame:
    if damping and geo == "none":
        raise ValueError("geo damping needs geo history")
    if geo == "none" and transitions == "none":
        return frame
    share = corpus_location_share(items)
    parts = [view_features(frame, v, items, share, geo, transitions, alpha, damping) for v in views]
    out = pl.concat(parts).sort("q", "rrf_rank")
    if out.height != frame.height:
        raise ValueError("history views must cover every query exactly once")
    return out
