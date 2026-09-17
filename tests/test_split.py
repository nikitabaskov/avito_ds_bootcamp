import polars as pl
import pytest

from candgen.core.data import assign_parts, build_contexts, prepare_queries, sample_eval_queries


def pairs() -> pl.DataFrame:
    return prepare_queries(
        pl.DataFrame(
            {
                "search_query": ["Ёлка  Москва", "елка москва", "елка москва", "баня", "баня"],
                "search_location_id": [1, 1, 2, 1, 1],
                "search_is_delivery_search": [0, 0, 0, 0, 0],
                "search_infm_params_text": ["", "", "", None, "Вид услуги"],
                "search_category": [114, 114, 114, 114, 114],
                "item_id": ["i1", "i1", "i2", "i3", "i3"],
            }
        )
    )


def test_normalized_texts_share_context_and_positives_are_sets():
    contexts = build_contexts(pairs())
    assert contexts.height == 4
    moscow = contexts.filter(
        (pl.col("query_text") == "елка москва") & (pl.col("search_location_id") == 1)
    )
    assert moscow["item_ids"].to_list() == [["i1"]]
    assert contexts["query_id"].n_unique() == 4
    assert contexts["query_id"].str.len_chars().unique().to_list() == [16]


def test_parts_are_deterministic_and_disjoint():
    texts = pl.Series([f"query {i}" for i in range(5000)])
    first = assign_parts(texts, seed=7)
    second = assign_parts(texts.shuffle(seed=1), seed=7)
    assert first.equals(second)
    assert first["query_text"].is_unique().all()
    shares = first["part"].value_counts(normalize=True).sort("part")
    assert shares["proportion"].to_list() == pytest.approx([0.1, 0.1, 0.8], abs=0.02)


def test_eval_sample_takes_one_context_per_text():
    sample = sample_eval_queries(build_contexts(pairs()), n=10, seed=3)
    assert sample["query_text"].is_unique().all()
    assert sample.height == 2
    assert sample.equals(sample_eval_queries(build_contexts(pairs()), n=10, seed=3))


def test_eval_sample_text_choice_ignores_context_count():
    def contexts(heavy_contexts: int) -> pl.DataFrame:
        texts = ["heavy"] * heavy_contexts + [f"text {i}" for i in range(200)]
        locations = list(range(heavy_contexts)) + [0] * 200
        return build_contexts(
            prepare_queries(
                pl.DataFrame(
                    {
                        "search_query": texts,
                        "search_location_id": locations,
                        "search_is_delivery_search": 0,
                        "search_infm_params_text": "",
                        "search_category": 114,
                        "item_id": "i",
                    }
                )
            )
        )

    for seed in range(20):
        light = sample_eval_queries(contexts(1), n=50, seed=seed)
        heavy = sample_eval_queries(contexts(300), n=50, seed=seed)
        assert light["query_text"].sort().equals(heavy["query_text"].sort())
        assert heavy["query_text"].is_unique().all()
