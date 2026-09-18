import numpy as np
import polars as pl
import pytest

from candgen.core.ranker import full_recall_curve


def test_full_recall_includes_missing_positives_empty_groups_and_sparse_ids():
    class Model:
        def predict(self, pool, ntree_end):
            return np.array([2.0, 1.0, 3.0])

    frame = pl.DataFrame(
        {"q": [2, 2, 5], "row": [0, 2, 2], "rrf_rank": [1, 2, 1], "score": [2.0, 1.0, 3.0]}
    )
    queries = pl.DataFrame({"q": [8, 2, 5], "item_ids": [["b"], ["a", "b"], ["b"]]})
    curve = full_recall_curve(Model(), frame, queries, ["a", "b", "c"], ["score"], [1], k=1)
    # q=2 retrieves half of its positives; q=5 misses, q=8 has no candidates.
    assert curve[1] == pytest.approx(1 / 6)
