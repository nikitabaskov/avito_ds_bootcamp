import numpy as np
import polars as pl
import pytest
import torch
from test_features import corpus
from test_history import contexts
from test_microcats import unit

from candgen.core.history import full_view, history_pairs, query_locations
from candgen.core.microcats import MicrocatIndex
from candgen.core.signals import (
    ENCODER_FEATURES,
    add_centroid_features,
    add_cosine_features,
    add_filter_features,
    parse_filters,
    query_centroids,
    region_hits,
)

ITEM_VECTORS = torch.from_numpy(unit([1, 0], [0, 1], [1, 1]))
ROWS = {"a": 0, "b": 1, "c": 2}


def centroid_frame(views) -> pl.DataFrame:
    frame = pl.DataFrame(
        {"q": pl.Series([0, 0, 0], dtype=pl.Int32), "row": [0, 1, 2], "rrf_rank": [1, 2, 3]}
    )
    index = MicrocatIndex(["баня", "ремонт"], unit([0, 1], [1, 0]), "cpu")
    centroids = query_centroids(views, index, unit([1, 0.1]), ITEM_VECTORS, ROWS)
    return add_centroid_features(frame, centroids, ITEM_VECTORS)


def test_centroid_follows_items_chosen_by_similar_texts():
    pairs = history_pairs(contexts(["ремонт", "баня"], [7, 7], [["a"], ["b"]]), corpus())
    queries = pl.DataFrame({"query_text": ["ремонт холодильника"], "search_location_id": [7]})
    out = centroid_frame(full_view(queries, pairs))
    assert out["nb_cos"][0] > out["nb_cos"][2] > out["nb_cos"][1]
    assert out["nb_cos_rank"].to_list() == [1.0, 3.0, 2.0]


def test_centroid_ignores_labels_outside_the_view():
    queries = pl.DataFrame({"query_text": ["ремонт холодильника"], "search_location_id": [7]})
    allowed = history_pairs(contexts(["баня"], [7], [["b"]]), corpus())
    held_out = [["a"], ["c"]]
    results = []
    for items in held_out:
        full = history_pairs(contexts(["ремонт", "баня"], [7, 7], [items, ["b"]]), corpus())
        views = [(query_locations(queries), allowed), (query_locations(queries).head(0), full)]
        results.append(centroid_frame(views)["nb_cos"].to_numpy())
    np.testing.assert_allclose(results[0], results[1])
    assert results[0][1] == pytest.approx(1.0)


def test_parse_filters_splits_known_keys():
    assert parse_filters("") == {}
    assert parse_filters("Вид услуги Красота, здоровье Тип услуги Маникюр") == {
        "Вид услуги": "Красота, здоровье",
        "Тип услуги": "Маникюр",
    }
    assert parse_filters("Тип услуги автосервиса Тюнинг") == {"Тип услуги автосервиса": "Тюнинг"}


def test_filter_features_match_item_params_and_keep_missing():
    frame = pl.DataFrame(
        {
            "q": pl.Series([0, 0, 0, 1], dtype=pl.Int32),
            "row": [0, 1, 2, 0],
            "rrf_rank": [1, 2, 3, 1],
        }
    )
    params = corpus()["item_infm_params_text"].fill_null("")
    out = add_filter_features(frame, ["Вид услуги Ремонт", ""], params)
    assert out["filt_vid"].to_list() == [1.0, 0.0, 0.0, None]
    assert out["filt_tip"].to_list() == [None] * 4
    assert out["filt_share"].to_list() == [1.0, 0.0, 0.0, None]


def test_region_hits_search_core_cities_of_uncentered_queries():
    pairs = history_pairs(
        contexts(["ремонт"] * 9 + ["баня"], [100] * 10, [["a"]] * 9 + [["c"]]), corpus()
    )
    queries = pl.DataFrame({"query_text": ["x", "y"], "search_location_id": [100, 7]})
    centroids = unit([1, 0], [1, 0])
    targets = np.array([True, False])
    rows, scores = region_hits(
        centroids,
        full_view(queries, pairs),
        queries["search_location_id"].to_numpy(),
        corpus()["item_location_id"].to_numpy(),
        targets,
        ITEM_VECTORS,
        2,
    )
    assert rows[0].tolist() == [0, 1]
    assert rows[1].tolist() == [-1, -1]
    assert scores[0][0] == pytest.approx(1.0)


def test_cosine_features_rank_items_within_each_query():
    frame = pl.DataFrame({"q": pl.Series([0, 0, 1, 1], dtype=pl.Int32), "row": [0, 1, 0, 2]})
    out = add_cosine_features(frame, tuple(ENCODER_FEATURES), unit([0, 1], [1, 0]), ITEM_VECTORS)
    assert out["enc2_sim"].to_list() == pytest.approx([0.0, 1.0, 1.0, 2**-0.5])
    assert out["enc2_rank"].to_list() == [2.0, 1.0, 1.0, 2.0]
