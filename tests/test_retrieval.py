import numpy as np
import pytest

from candgen.core.bm25 import BM25Config, BM25Retriever
from candgen.core.retrieval import order_hits, rrf_fuse, search_local


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
