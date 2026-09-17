import numpy as np
import polars as pl

from candgen.core.metrics import recall_at_k

POOL_KS = (50, 100, 300, 1000)


def recall_report(queries: pl.DataFrame, candidates: list[list[str]], seen_items: set[str]) -> dict:
    relevant = queries["item_ids"].to_list()
    per_k = {k: recall_at_k(candidates, relevant, k) for k in POOL_KS}
    r50 = per_k[50]

    words = queries["query_text"].str.split(" ").list.len()
    masks = {
        "filters_empty": (queries["search_infm_params_text"] == "").to_numpy(),
        "filters_set": (queries["search_infm_params_text"] != "").to_numpy(),
        "category_0": (queries["search_category"] == 0).to_numpy(),
        "words_1": (words == 1).to_numpy(),
        "words_2": (words == 2).to_numpy(),
        "words_3": (words == 3).to_numpy(),
        "words_4plus": (words >= 4).to_numpy(),
    }
    slices = {
        name: {"n": int(m.sum()), "recall@50": float(r50[m].mean()) if m.any() else None}
        for name, m in masks.items()
    }

    new_rel = [[i for i in rel if i not in seen_items] for rel in relevant]
    has_new = np.array([bool(r) for r in new_rel])
    new_r50 = recall_at_k(
        [c for c, h in zip(candidates, has_new, strict=True) if h],
        [r for r in new_rel if r],
        50,
    )
    slices["new_items"] = {
        "n": int(has_new.sum()),
        "recall@50": float(new_r50.mean()) if has_new.any() else None,
    }

    return {
        "queries": queries.height,
        "recall": {f"@{k}": float(v.mean()) for k, v in per_k.items()},
        "mean_pool_size": float(np.mean([len(c) for c in candidates])),
        "empty_pool_queries": int(sum(not c for c in candidates)),
        "slices": slices,
    }
