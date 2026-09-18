import numpy as np
import polars as pl
import pytest
from test_features import corpus
from test_history import contexts

from candgen.core.history import full_view, history_pairs
from candgen.core.microcats import MicrocatIndex, add_microcat_features, text_microcats


def unit(*rows: list[float]) -> np.ndarray:
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_text_microcats_normalize_each_text():
    pairs = history_pairs(
        contexts(["x", "x", "y"], [7, 9, 7], [["a", "b"], ["a"], ["c"]]), corpus()
    )
    shares = {
        (r["query_text"], r["item_microcat_id"]): r["share"]
        for r in text_microcats(pairs).iter_rows(named=True)
    }
    assert shares == {("x", 1): pytest.approx(2 / 3), ("x", 2): pytest.approx(1 / 3), ("y", 3): 1.0}


def test_neighbor_microcats_score_candidates_and_mark_unknown():
    pairs = history_pairs(contexts(["ремонт", "баня"], [7, 7], [["a"], ["b"]]), corpus())
    index = MicrocatIndex(["баня", "ремонт", "чужой"], unit([0, 1], [1, 0], [1, 1]), "cpu")
    queries = pl.DataFrame({"query_text": ["ремонт холодильника"], "search_location_id": [7]})
    frame = pl.DataFrame(
        {"q": pl.Series([0, 0, 0], dtype=pl.Int32), "row": [0, 1, 2], "rrf_rank": [1, 2, 3]}
    )
    out = add_microcat_features(
        frame, full_view(queries, pairs), index, unit([1, 0.1]), corpus()["item_microcat_id"]
    )
    probs = out["mc_prob"].to_list()
    assert probs[0] > 0.99 and probs[1] < 0.01 and probs[2] == 0.0
    assert out["mc_ratio"][0] == pytest.approx(1.0)
    assert out["mc_top_sim"][0] == pytest.approx(float(unit([1, 0.1])[0, 0]))
    assert out.columns == [*frame.columns, "mc_prob", "mc_ratio", "mc_entropy", "mc_top_sim"]
