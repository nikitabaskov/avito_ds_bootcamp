import numpy as np
import pytest

from candgen.core.bm25 import BM25Config, BM25Retriever
from candgen.core.retrieval import (
    geo_rerank,
    order_hits,
    radius_groups,
    rrf_fuse,
    search_groups,
    search_local,
)


def test_order_hits_breaks_ties_by_row():
    rows = np.array([[5, 2, 9, 1]])
    scores = np.array([[0.5, 0.9, 0.5, -np.inf]])
    ordered_rows, ordered_scores = order_hits(rows, scores)
    assert ordered_rows.tolist() == [[2, 5, 9, 1]]
    assert ordered_scores[0, 0] == pytest.approx(0.9)


def test_rrf_sums_reciprocal_ranks_and_skips_padding():
    first = np.array([[10, 20, -1]])
    second = np.array([[20, 30, -1]])
    fused = rrf_fuse([first, second], k=60)
    assert fused[0].tolist() == [20, 10, 30]


def test_rrf_equal_scores_ordered_by_row():
    fused = rrf_fuse([np.array([[7]]), np.array([[3]])], k=60)
    assert fused[0].tolist() == [3, 7]


def test_rrf_rejects_mismatched_runs():
    with pytest.raises(ValueError):
        rrf_fuse([np.array([[1]]), np.array([[1], [2]])])


def test_search_local_restricts_items_and_pads_missing_locations():
    item_locations = np.array([1, 2, 1, 1])
    query_locations = np.array([1, 3, 2])
    seen = []

    def search(queries, items, depth):
        seen.append((queries.tolist(), items.tolist(), depth))
        rows = np.tile(items[::-1][:depth], (len(queries), 1))
        return rows, np.ones(rows.shape, dtype=np.float32)

    rows, _ = search_local(search, query_locations, item_locations, k=2)
    assert rows.tolist() == [[3, 2], [-1, -1], [1, -1]]
    assert seen == [([0], [0, 2, 3], 2), ([2], [1], 1)]


def test_bm25_local_search_returns_only_same_location_matches():
    documents = ["ремонт холодильников", "ремонт стиральных машин", "ремонт холодильника дома"]
    retriever = BM25Retriever(BM25Config())
    retriever.index(documents)
    rows, _ = retriever.search_local(
        ["ремонт холодильника"], np.array([7]), np.array([1, 7, 7]), k=3
    )
    assert rows.tolist() == [[2, 1, -1]]


def test_radius_groups_share_centers_and_skip_unknown():
    centers = np.array([[55.0, 37.0], [np.nan, np.nan], [55.0, 37.0], [56.0, 37.0]])
    items = np.array([[55.0, 37.0], [55.1, 37.0], [56.0, 37.0], [np.nan, np.nan]])
    groups = radius_groups(centers, items, radius_km=20.0)
    assert sorted((q.tolist(), i.tolist()) for q, i in groups) == [
        ([0, 2], [0, 1]),
        ([3], [2]),
    ]


def test_search_groups_pads_queries_outside_groups():
    groups = [(np.array([2]), np.array([4, 5])), (np.array([0]), np.array([], dtype=np.int64))]

    def search(queries, items, depth):
        return np.tile(items[:depth], (len(queries), 1)), np.ones((len(queries), depth))

    rows, _ = search_groups(search, groups, n_queries=3, k=3)
    assert rows.tolist() == [[-1, -1, -1], [-1, -1, -1], [4, 5, -1]]


def test_geo_rerank_penalizes_other_locations_only():
    rows = np.array([[0, 1, 2, -1]])
    scores = np.array([[10.0, 9.0, 8.0, -np.inf]], dtype=np.float32)
    item_locations = np.array([7, 5, 5])
    ordered, adjusted = geo_rerank((rows, scores), np.array([5]), item_locations, 0.6, 0.0)
    assert ordered.tolist() == [[1, 2, 0, -1]]
    assert adjusted[0, 2] == pytest.approx(6.0)
    ordered, _ = geo_rerank((rows, scores), np.array([5]), item_locations, 1.0, 1.5)
    assert ordered.tolist() == [[1, 0, 2, -1]]
