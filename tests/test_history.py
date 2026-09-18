import numpy as np
import polars as pl
import pytest
from test_features import corpus

from candgen.core.features import ItemTable
from candgen.core.history import (
    add_history_features,
    crossfit_views,
    full_view,
    history_pairs,
    location_centers,
    query_centers,
    text_fold,
)


def contexts(texts: list[str], locations: list[int], items: list[list[str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"query_text": texts, "search_location_id": locations, "item_ids": items},
        schema={
            "query_text": pl.String,
            "search_location_id": pl.Int64,
            "item_ids": pl.List(pl.String),
        },
    )


def test_location_centers_use_unique_pairs_with_coordinates():
    pairs = history_pairs(
        contexts(["a", "b", "c"], [100, 100, 7], [["a", "c"], ["c"], []]), corpus()
    )
    centers = {r["search_location_id"]: r for r in location_centers(pairs).iter_rows(named=True)}
    assert set(centers) == {100}
    assert centers[100]["hist_pairs"] == 3.0 and centers[100]["hist_items"] == 2.0
    assert centers[100]["hist_lat"] == pytest.approx(56.0)


def centers_by_query(queries: pl.DataFrame, views) -> pl.DataFrame:
    return pl.concat(
        q.join(location_centers(pairs), on="search_location_id", how="left") for q, pairs in views
    ).sort("q")


def test_crossfit_views_ignore_own_fold_and_holdout_uses_full_history():
    texts = [f"text {i}" for i in range(40)]
    folds = text_fold(pl.Series(texts)).to_list()
    target = next(i for i, f in enumerate(folds) if f == 0)
    other = next(i for i, f in enumerate(folds) if f != 0)
    holdout = pl.DataFrame({"q": pl.Series([other], dtype=pl.Int32)})
    queries = contexts(texts, [100] * 40, [["a"]] * 40)
    changed = [["c"] if f == 0 else ["a"] for f in folds]

    base_views = crossfit_views(queries, history_pairs(queries, corpus()), holdout)
    moved_views = crossfit_views(
        queries, history_pairs(contexts(texts, [100] * 40, changed), corpus()), holdout
    )
    assert sum(v[0].height for v in base_views) == 40
    base, moved = centers_by_query(queries, base_views), centers_by_query(queries, moved_views)
    assert base.row(target) == moved.row(target)
    assert base.row(other) != moved.row(other)
    assert moved.row(other, named=True)["hist_items"] == 2.0


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "q": pl.Series([0, 1, 2], dtype=pl.Int32),
            "row": pl.Series([0, 2, 2], dtype=pl.Int64),
            "rrf_rank": [1.0, 1.0, 1.0],
            "dist_km": pl.Series([5.0, None, None], dtype=pl.Float32),
            "location_items": [2.0, 0.0, 0.0],
        }
    )


def test_geo_fallback_fills_distance_only_without_corpus_center():
    queries = contexts(["x", "y", "z"], [7, 100, 555], [[], [], []])
    views = full_view(queries, history_pairs(contexts(["h"], [100], [["a"]]), corpus()))
    items = ItemTable(corpus())

    out = add_history_features(frame(), views, items, "full", "none", 10.0)
    assert out["dist_km"][0] == pytest.approx(5.0)
    assert out["dist_km"][1] == pytest.approx(111.2, abs=0.5)
    assert out["dist_km"][2] is None
    assert out["center_source"].to_list() == [0.0, 1.0, 2.0]
    assert out["hist_pairs"].to_list() == [0.0, 1.0, 0.0]
    assert out["hist_dist_km"][0] is None and out["hist_spread_km"][1] == pytest.approx(0.0)
    assert add_history_features(frame(), views, items, "none", "none", 10.0).equals(frame())


def test_transitions_smooth_towards_corpus_share():
    queries = contexts(["x", "y", "z"], [7, 7, 555], [[], [], []])
    history = contexts(["h1", "h2", "h3"], [7, 7, 7], [["a"], ["b"], ["c"]])
    views = full_view(queries, history_pairs(history, corpus()))
    items = ItemTable(corpus())

    out = add_history_features(frame(), views, items, "none", "full", 3.0)
    by_q = {r["q"]: r for r in out.iter_rows(named=True)}
    assert by_q[0]["trans_pairs"] == 2.0 and by_q[0]["trans_support"] == 3.0
    assert by_q[0]["trans_prob"] == pytest.approx((2 + 3 * 2 / 3) / 6)
    assert by_q[1]["trans_prob"] == pytest.approx((1 + 3 * 1 / 3) / 6)
    assert by_q[1]["trans_targets"] == 2.0 and by_q[1]["trans_outside_share"] == 0.0
    assert by_q[2]["trans_unknown"] == 1.0 and by_q[2]["trans_prob"] == pytest.approx(1 / 3)
    assert by_q[2]["trans_lift"] == pytest.approx(np.log(2.0))
    assert out["dist_km"].to_list() == frame()["dist_km"].to_list()


def test_query_centers_prefer_corpus_then_history():
    queries = contexts(["x", "y", "z"], [7, 100, 555], [[], [], []])
    views = full_view(queries, history_pairs(contexts(["h"], [100], [["c"]]), corpus()))
    centers = query_centers(views, ItemTable(corpus()))
    assert centers[0].tolist() == [55.0, 37.0]
    assert centers[1].tolist() == [56.0, 37.0]
    assert np.isnan(centers[2]).all()
