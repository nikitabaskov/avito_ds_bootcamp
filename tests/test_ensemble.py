import polars as pl

from candgen.core.ensemble import rank_fusion, union_pool


def ranks(q: list[int], rows: list[int], model_rank: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"q": q, "row": rows, "model_rank": model_rank},
        schema={"q": pl.Int32, "row": pl.Int64, "model_rank": pl.Int32},
    )


def test_rank_fusion_sums_reciprocal_ranks_and_breaks_ties_by_row():
    a = ranks([0, 0, 0, 2], [10, 11, 12, 5], [1, 2, 3, 1])
    b = ranks([0, 0, 0], [12, 11, 13], [1, 2, 3])
    fused = rank_fusion([a, b], 3, top=3)
    assert [f.tolist() for f in fused] == [[12, 11, 10], [], [5]]
    tie = rank_fusion([ranks([0, 0], [2, 1], [1, 2]), ranks([0, 0], [1, 2], [1, 2])], 1)
    assert tie[0].tolist() == [1, 2]


def test_single_model_fusion_keeps_model_order_and_union_covers_all_rows():
    a = ranks([0, 0, 0], [7, 3, 9], [1, 2, 3])
    assert rank_fusion([a], 1, top=2)[0].tolist() == [7, 3]
    b = ranks([0], [4], [1])
    assert sorted(union_pool([a, b], 1)[0].tolist()) == [3, 4, 7, 9]
