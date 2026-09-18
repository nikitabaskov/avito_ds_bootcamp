import numpy as np
import polars as pl
import pytest
import torch

from candgen.core.data import prepare_queries
from candgen.core.features import (
    FEATURES,
    ItemTable,
    attach_labels,
    build_features,
    candidate_pool,
)


def hits(rows):
    rows = np.array(rows, dtype=np.int64)
    return rows, np.where(rows >= 0, 10.0 - np.arange(rows.shape[1]), -np.inf).astype(np.float32)


def corpus() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "item_title_raw": ["Ремонт холодильников", "Баня на дровах", None],
            "item_description_raw": ["", None, "текст"],
            "item_infm_params_text": ["Вид услуги Ремонт", "", None],
            "item_category_id": [114, 114, 5],
            "item_microcat_id": [1, 2, 3],
            "item_price": [100.0, -1.0, None],
            "item_rating": [5.0, None, 4.5],
            "item_rating_reviews_count": [3, None, 1],
            "item_location_id": [7, 7, 9],
            "item_latitude": [55.0, 55.0, 56.0],
            "item_longitude": [37.0, 37.0, 37.0],
            "item_is_phone_hidden": [0, 1, 0],
            "item_is_message_forbidden": [0, 0, 1],
        }
    )


def test_candidate_pool_merges_lists_and_orders_by_rrf():
    runs = {
        "bm25_global": hits([[0, 1, -1]]),
        "bm25_local": hits([[-1, -1, -1]]),
        "dense_global": hits([[1, 2, 0]]),
        "dense_local": hits([[1, -1, -1]]),
    }
    depths = {"bm25_global": 3, "bm25_local": 3, "dense_global": 2, "dense_local": 3}
    pool = candidate_pool(runs, depths, rrf_k=60)
    assert pool["row"].to_list() == [1, 0, 2]
    assert pool["n_lists"].to_list() == [3, 1, 1]
    assert pool["rrf_rank"].to_list() == [1, 2, 3]
    assert pool["rank_bm25_local"].is_null().all()
    assert pool["rrf"][0] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


def test_build_features_pairs_query_and_item_signals():
    queries = prepare_queries(
        pl.DataFrame(
            {
                "search_query": ["ремонт холодильника", "баня"],
                "search_location_id": [7, 100],
                "search_is_delivery_search": [0, 0],
                "search_infm_params_text": ["", "Вид услуги"],
                "search_category": [114, 0],
            }
        )
    )
    runs = {name: hits([[0, 1], [2, -1]]) for name in ("bm25_global", "dense_global")}
    runs |= {name: hits([[-1, -1], [-1, -1]]) for name in ("bm25_local", "dense_local")}
    pool = candidate_pool(runs, dict.fromkeys(runs, 2), rrf_k=60)
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    frame = build_features(
        pool, queries, ItemTable(corpus()), np.array([[1.0, 0.0], [0.0, 1.0]]), vectors
    )
    assert frame.columns == ["q", "row", *FEATURES]
    by_pair = {(r["q"], r["row"]): r for r in frame.iter_rows(named=True)}
    first = by_pair[(0, 0)]
    assert first["title_overlap"] == pytest.approx(1.0)
    assert first["params_overlap"] == pytest.approx(0.5)
    assert first["loc_match"] == 1.0
    assert first["dist_km"] == pytest.approx(0.0)
    assert first["location_items"] == 2.0
    assert first["dense_sim"] == pytest.approx(1.0)
    assert by_pair[(0, 1)]["dense_sim"] == pytest.approx(0.0)
    other = by_pair[(1, 2)]
    assert other["dense_sim"] == pytest.approx(0.8)
    assert other["query_cat_zero"] == 1.0 and other["filters_empty"] == 0.0
    assert other["dist_km"] is None and other["location_items"] == 0.0
    assert other["cat_match"] == 0.0 and other["price"] is None

    labelled = attach_labels(frame, [["b", "zzz"], []], corpus()["item_id"].to_list())
    assert dict(zip(labelled["row"], labelled["label"], strict=True)) == {0: 0, 1: 1, 2: 0}


def test_radius_lists_add_their_own_columns():
    runs = {name: hits([[0, 1]]) for name in ("bm25_global", "dense_global")}
    runs["bm25_radius"] = hits([[2, -1]])
    pool = candidate_pool(runs, dict.fromkeys(runs, 2), rrf_k=60)
    assert pool["rank_bm25_radius"].to_list()[pool["row"].to_list().index(2)] == 1.0
    assert pool["rank_bm25_local"].is_null().all()
    queries = prepare_queries(
        pl.DataFrame(
            {
                "search_query": ["баня"],
                "search_location_id": [7],
                "search_is_delivery_search": [0],
                "search_infm_params_text": [""],
                "search_category": [114],
            }
        )
    )
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    frame = build_features(pool, queries, ItemTable(corpus()), np.array([[1.0, 0.0]]), vectors)
    assert frame.columns == ["q", "row", *FEATURES, "rank_bm25_radius", "score_bm25_radius"]
