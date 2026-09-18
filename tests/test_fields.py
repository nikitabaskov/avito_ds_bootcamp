import polars as pl
from test_features import corpus

from candgen.core.data import prepare_queries
from candgen.core.fields import FIELD_FEATURES, FieldScorer


def test_field_scores_split_title_params_and_filters():
    queries = prepare_queries(
        pl.DataFrame(
            {
                "search_query": ["ремонт холодильника", "баня"],
                "search_location_id": [7, 7],
                "search_is_delivery_search": [0, 0],
                "search_infm_params_text": ["Вид услуги Ремонт", ""],
                "search_category": [114, 114],
            }
        )
    )
    frame = pl.DataFrame(
        {
            "q": pl.Series([0, 0, 1], dtype=pl.Int32),
            "row": pl.Series([0, 1, 1], dtype=pl.Int64),
            "rrf_rank": [1.0, 2.0, 1.0],
        }
    )
    out = FieldScorer(corpus(), "filters").add(frame, queries)
    assert out.columns == ["q", "row", "rrf_rank", *FIELD_FEATURES["filters"]]
    rows = {(r["q"], r["row"]): r for r in out.iter_rows(named=True)}
    assert rows[(0, 0)]["bm25_title"] > 0 and rows[(0, 1)]["bm25_title"] == 0
    assert rows[(0, 0)]["bm25_params"] > 0 and rows[(0, 0)]["bm25_filters"] > 0
    assert rows[(1, 1)]["bm25_title"] > 0 and rows[(1, 1)]["bm25_filters"] == 0
    assert rows[(1, 1)]["bm25_params"] == 0 and rows[(0, 1)]["bm25_description"] == 0
    assert FieldScorer(corpus(), "none").add(frame, queries).equals(frame)
