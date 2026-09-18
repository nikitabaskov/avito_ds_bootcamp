import polars as pl
import pytest

from candgen.core.evaluation import average_seed_metrics, compare_per_query


def test_compare_per_query_reports_verdicts_and_slices():
    base = pl.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(40)],
            "recall": [0.0] * 40,
            "pool_recall": [1.0] * 40,
            "has_center": [i < 30 for i in range(40)],
            "filters_set": [i % 2 == 0 for i in range(40)],
            "words": ["1" if i < 5 else "2" for i in range(40)],
            "any_other_location": [i >= 35 for i in range(40)],
        }
    )
    better = base.with_columns(recall=pl.Series([1.0] * 30 + [0.0] * 10))
    result = compare_per_query(better, base.sample(fraction=1.0, shuffle=True, seed=1))
    assert result["verdict"] == "better" and result["diff"] == pytest.approx(0.75)
    assert result["pool_recall"]["verdict"] == "undetermined"
    assert result["slices"]["has_center"]["verdict"] == "better"
    no_center = result["slices"]["no_center"]
    assert no_center["n"] == 10 and no_center["diff"] == 0.0
    with pytest.raises(ValueError):
        compare_per_query(better, base.head(39))


def test_seed_metrics_align_queries_and_reject_context_changes():
    first = pl.DataFrame(
        {
            "query_id": ["a", "b"],
            "recall": [1.0, 0.0],
            "pool_recall": [1.0, 1.0],
            "has_center": [True, False],
        }
    )
    second = first.reverse().with_columns(recall=pl.Series([0.5, 0.0]))
    mean = average_seed_metrics([first, second])
    assert mean["recall"].to_list() == [0.5, 0.25]
    with pytest.raises(ValueError, match="different queries or contexts"):
        average_seed_metrics([first, second.with_columns(has_center=pl.lit(True))])
    with pytest.raises(ValueError, match="duplicate query_id"):
        average_seed_metrics([pl.concat([first, first])])
