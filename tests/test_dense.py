import numpy as np
import polars as pl

from candgen.core.dense import DenseIndex, passage_texts, query_texts


def test_passage_puts_title_first_and_truncates_params():
    items = pl.DataFrame(
        {
            "item_title_raw": ["Ремонт  холодильников"],
            "item_infm_params_text": ["Вид услуги Ремонт техники"],
            "item_description_raw": [None],
        }
    )
    assert passage_texts(items, params_chars=10) == ["passage: Ремонт холодильников\nВид услуги"]


def test_query_appends_filters_only_when_present():
    queries = pl.DataFrame(
        {"query_text": ["баня", "маникюр"], "search_infm_params_text": ["", "Вид услуги Красота"]}
    )
    assert query_texts(queries, with_filters=True) == [
        "query: баня",
        "query: маникюр. Вид услуги Красота",
    ]
    assert query_texts(queries, with_filters=False) == ["query: баня", "query: маникюр"]


def test_dense_index_global_and_local_search():
    embeddings = np.array([[1, 0], [0.8, 0.6], [0, 1], [0.6, 0.8]], dtype=np.float32)
    index = DenseIndex(embeddings, device="cpu", block_size=1)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)

    rows, scores = index.search(queries, k=2)
    assert rows.tolist() == [[0, 1], [2, 3]]
    assert scores[0, 0] >= scores[0, 1]

    rows, _ = index.search_local(queries, np.array([5, 5]), np.array([9, 5, 9, 5]), k=3)
    assert rows.tolist() == [[1, 3, -1], [3, 1, -1]]
