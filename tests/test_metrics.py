import numpy as np
import pytest

from candgen.core.evaluation import paired_bootstrap
from candgen.core.metrics import recall_at_k


def test_assignment_example():
    predictions = [["a", "x"], ["b", "y"], ["z"]]
    relevant = [{"a"}, {"b", "c"}, {"d"}]
    assert recall_at_k(predictions, relevant, 50).mean() == pytest.approx(0.5)


def test_cutoff_and_duplicate_predictions():
    predictions = [["x", "a", "a", "b"]]
    relevant = [["a", "b"]]
    np.testing.assert_allclose(recall_at_k(predictions, relevant, 2), [0.5])
    np.testing.assert_allclose(recall_at_k(predictions, relevant, 3), [0.5])
    np.testing.assert_allclose(recall_at_k(predictions, relevant, 4), [1.0])


def test_repeated_positive_rows_count_once():
    np.testing.assert_allclose(recall_at_k([["a"]], [["a", "a", "b"]], 50), [0.5])


def test_full_miss_and_empty_prediction():
    np.testing.assert_allclose(recall_at_k([["x"], []], [{"a"}, {"b"}], 50), [0.0, 0.0])


def test_rejects_empty_relevant_and_length_mismatch():
    with pytest.raises(ValueError):
        recall_at_k([["a"]], [set()], 50)
    with pytest.raises(ValueError):
        recall_at_k([["a"]], [{"a"}, {"b"}], 50)


def test_paired_bootstrap_ci_contains_mean_difference():
    rng = np.random.default_rng(0)
    base = rng.random(500)
    result = paired_bootstrap(base + 0.1, base, n=200)
    assert result["diff"] == pytest.approx(0.1)
    assert result["ci95"][0] == pytest.approx(0.1) and result["ci95"][1] == pytest.approx(0.1)
