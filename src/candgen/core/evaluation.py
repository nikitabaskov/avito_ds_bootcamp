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


def paired_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray, n: int = 5000, seed: int = 42
) -> dict:
    diff = candidate - baseline
    idx = np.random.default_rng(seed).integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "diff": float(diff.mean()),
        "ci95": [float(low), float(high)],
        "resamples": n,
        "verdict": verdict(float(low), float(high)),
    }


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "undetermined"


COMPARISON_SLICES = {
    "no_center": ~pl.col("has_center"),
    "has_center": pl.col("has_center"),
    "filters_set": pl.col("filters_set"),
    "filters_empty": ~pl.col("filters_set"),
    "words_1": pl.col("words") == "1",
    "other_location": pl.col("any_other_location"),
}


def compare_per_query(per_query: pl.DataFrame, reference: pl.DataFrame) -> dict:
    joined = per_query.join(
        reference.select("query_id", ref_recall="recall", ref_pool="pool_recall"),
        on="query_id",
        how="full",
    )
    if joined["query_id"].null_count() or joined["query_id_right"].null_count():
        raise ValueError("runs were evaluated on different queries")
    slices = {}
    for name, mask in COMPARISON_SLICES.items():
        part = joined.filter(mask)
        slices[name] = {
            "n": part.height,
            "recall@50": float(part["recall"].mean()),
            "reference_recall@50": float(part["ref_recall"].mean()),
            **paired_bootstrap(part["recall"].to_numpy(), part["ref_recall"].to_numpy()),
        }
    return {
        "reference_recall@50": float(joined["ref_recall"].mean()),
        **paired_bootstrap(joined["recall"].to_numpy(), joined["ref_recall"].to_numpy()),
        "pool_recall": paired_bootstrap(
            joined["pool_recall"].to_numpy(), joined["ref_pool"].to_numpy()
        ),
        "slices": slices,
    }
