from collections.abc import Sequence

import numpy as np
import polars as pl

RRF_K = 60.0


def rank_fusion(
    ranks: Sequence[pl.DataFrame], n_queries: int, k: float = RRF_K, top: int = 50
) -> list[np.ndarray]:
    fused = (
        pl.concat([r.select("q", "row", w=1.0 / (k + pl.col("model_rank"))) for r in ranks])
        .group_by("q", "row")
        .agg(score=pl.col("w").sum())
        .sort(["q", "score", "row"], descending=[False, True, False])
        .group_by("q", maintain_order=True)
        .head(top)
        .group_by("q", maintain_order=True)
        .agg("row")
    )
    out = [np.empty(0, dtype=np.int64) for _ in range(n_queries)]
    for q, rows in fused.iter_rows():
        out[q] = np.array(rows, dtype=np.int64)
    return out


def union_pool(ranks: Sequence[pl.DataFrame], n_queries: int) -> list[np.ndarray]:
    return rank_fusion(ranks, n_queries, top=np.iinfo(np.int32).max)
