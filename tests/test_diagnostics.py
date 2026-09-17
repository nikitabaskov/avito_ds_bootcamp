import numpy as np
import polars as pl
import pytest

from candgen.core.diagnostics import error_map, model_ranks, query_outcomes


def positives() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "q": [0, 1, 1, 2],
            "query_id": ["a", "b", "b", "c"],
            "item_id": ["x", "y", "z", "w"],
            "n_pos": [1, 2, 2, 1],
            "words": ["1", "2", "2", "3"],
            "filters_set": [False, True, True, False],
            "has_center": [True, True, True, False],
            "same_location": [True, False, True, None],
            "item_seen": [True, False, False, False],
            "n_lists": [4, 1, None, 2],
            "model_rank": [3, 80, None, 400],
            "in_pool": [True, True, False, True],
            "in_top50": [True, False, False, False],
            "dist_km": [1.0, 20.0, None, None],
            "filter_overlap": [None, 0.5, 1.0, None],
        }
    )


def test_model_ranks_order_by_score_then_rrf_rank():
    frame = pl.DataFrame({"q": [0, 0, 0, 1], "row": [5, 6, 7, 8], "rrf_rank": [1.0, 2.0, 3.0, 1.0]})
    ranks = model_ranks(frame, np.array([0.1, 0.9, 0.9, 0.0]))
    assert ranks["row"].to_list() == [6, 7, 5, 8]
    assert ranks["model_rank"].to_list() == [1, 2, 3, 1]


def test_query_outcomes_average_positives_per_query():
    per_query = query_outcomes(positives())
    assert per_query["recall"].to_list() == [1.0, 0.0, 0.0]
    assert per_query["pool_recall"].to_list() == [1.0, 0.5, 1.0]
    assert per_query["n_pos"].to_list() == [1, 2, 1]


def test_error_map_losses_sum_to_one_minus_recall():
    result = error_map(positives())
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["retrieval_loss"] == pytest.approx(1 / 6)
    assert result["selection_loss"] == pytest.approx(1 / 2)
    assert result["queries_with_retrieval_miss"] == 1
    assert result["queries_with_selection_miss"] == 2
    ranks = {r["group"]: r for r in result["by"]["model_rank"]}
    assert set(ranks) == {"1-50", "51-100", ">300", "not_in_pool"}
    assert ranks[">300"]["selection_loss"] == pytest.approx(1 / 3)
    for rows in result["by"].values():
        total = sum(r["retrieval_loss"] + r["selection_loss"] for r in rows)
        assert total == pytest.approx(2 / 3)
