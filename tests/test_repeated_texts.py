import polars as pl

from candgen.core.data import repeated_text_queries


def test_repeated_text_queries_need_other_contexts_and_skip_excluded_texts():
    contexts = pl.DataFrame(
        {
            "query_text": ["a", "a", "b", "c", "c", "d", "d"],
            "query_id": [f"q{i}" for i in range(7)],
        }
    )
    picked = repeated_text_queries(contexts, pl.Series(["d"]), n=10)
    assert sorted(picked["query_text"].to_list()) == ["a", "c"]
    assert picked["query_id"].n_unique() == 2
    assert repeated_text_queries(contexts, pl.Series(["d"]), n=1).height == 1
