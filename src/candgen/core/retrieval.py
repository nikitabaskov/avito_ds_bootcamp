from collections.abc import Callable, Sequence

import numpy as np

Hits = tuple[np.ndarray, np.ndarray]
GroupSearch = Callable[[np.ndarray, np.ndarray, int], Hits]


def order_hits(rows: np.ndarray, scores: np.ndarray) -> Hits:
    order = np.lexsort((rows, -scores), axis=1)
    return np.take_along_axis(rows, order, axis=1), np.take_along_axis(scores, order, axis=1)


def search_local(
    search: GroupSearch, query_locations: np.ndarray, item_locations: np.ndarray, k: int
) -> Hits:
    rows = np.full((len(query_locations), k), -1, dtype=np.int64)
    scores = np.full((len(query_locations), k), -np.inf, dtype=np.float32)
    for location in np.unique(query_locations):
        items = np.flatnonzero(item_locations == location)
        if items.size == 0:
            continue
        queries = np.flatnonzero(query_locations == location)
        group_rows, group_scores = search(queries, items, min(k, items.size))
        rows[queries, : group_rows.shape[1]] = group_rows
        scores[queries, : group_scores.shape[1]] = group_scores
    return rows, scores


def rrf_fuse(runs: Sequence[np.ndarray], k: int = 60) -> list[np.ndarray]:
    sizes = {run.shape[0] for run in runs}
    if len(sizes) != 1:
        raise ValueError(f"runs have different query counts: {sorted(sizes)}")
    fused = []
    for q in range(sizes.pop()):
        acc: dict[int, float] = {}
        for run in runs:
            for rank, row in enumerate(run[q].tolist(), start=1):
                if row >= 0:
                    acc[row] = acc.get(row, 0.0) + 1.0 / (k + rank)
        fused.append(np.array(sorted(acc, key=lambda r: (-acc[r], r)), dtype=np.int64))
    return fused
