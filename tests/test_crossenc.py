import polars as pl

from candgen.core.crossenc import add_cross_encoder_features, scored_pairs


def pool() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "q": pl.Series([0, 0, 0, 1, 1], dtype=pl.Int32),
            "row": [5, 3, 9, 3, 4],
            "rrf_rank": pl.Series([1, 2, 3, 1, 2], dtype=pl.Float32),
        }
    )


def test_scored_pairs_keep_top_rrf_ranks_only():
    assert scored_pairs(pool(), 2).rows() == [(0, 3), (0, 5), (1, 3), (1, 4)]


def test_features_rank_within_query_and_leave_unscored_null():
    frame = pool()
    scored = scored_pairs(frame, 2).with_columns(ce_score=pl.Series([0.2, 0.9, -1.0, 0.5]))
    out = add_cross_encoder_features(frame, scored)
    assert out.columns == [*frame.columns, "ce_score", "ce_rank", "ce_gap"]
    assert out.select("q", "row").rows() == frame.select("q", "row").rows()
    by_pair = {(r["q"], r["row"]): r for r in out.iter_rows(named=True)}
    assert by_pair[(0, 5)]["ce_rank"] == 1 and by_pair[(0, 3)]["ce_rank"] == 2
    assert abs(by_pair[(0, 3)]["ce_gap"] - (0.2 - 0.9)) < 1e-6
    assert by_pair[(0, 9)]["ce_score"] is None and by_pair[(0, 9)]["ce_rank"] is None
    assert by_pair[(1, 4)]["ce_rank"] == 1 and by_pair[(1, 4)]["ce_gap"] == 0.0
