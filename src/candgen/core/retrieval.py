"""Групповой поиск, географические ограничения списков и RRF."""

from collections.abc import Callable, Sequence

import numpy as np

Hits = tuple[np.ndarray, np.ndarray]
GroupSearch = Callable[[np.ndarray, np.ndarray, int], Hits]
Groups = list[tuple[np.ndarray, np.ndarray]]
EARTH_RADIUS_KM = 6371.0


def order_hits(rows: np.ndarray, scores: np.ndarray) -> Hits:
    order = np.lexsort((rows, -scores), axis=1)
    return np.take_along_axis(rows, order, axis=1), np.take_along_axis(scores, order, axis=1)


def search_groups(search: GroupSearch, groups: Groups, n_queries: int, k: int) -> Hits:
    rows = np.full((n_queries, k), -1, dtype=np.int64)
    scores = np.full((n_queries, k), -np.inf, dtype=np.float32)
    for queries, items in groups:
        if items.size == 0:
            continue
        group_rows, group_scores = search(queries, items, min(k, items.size))
        rows[queries, : group_rows.shape[1]] = group_rows
        scores[queries, : group_scores.shape[1]] = group_scores
    return rows, scores


def geo_rerank(
    hits: Hits,
    query_locations: np.ndarray,
    item_locations: np.ndarray,
    weight: float,
    delta: float,
) -> Hits:
    rows, scores = hits
    other = (rows >= 0) & (item_locations[np.maximum(rows, 0)] != query_locations[:, None])
    return order_hits(rows, np.where(other, scores * weight - delta, scores).astype(np.float32))


def location_groups(query_locations: np.ndarray, item_locations: np.ndarray) -> Groups:
    return [
        (np.flatnonzero(query_locations == location), np.flatnonzero(item_locations == location))
        for location in np.unique(query_locations)
    ]


def radius_groups(centers: np.ndarray, item_coords: np.ndarray, radius_km: float) -> Groups:
    located = np.flatnonzero(~np.isnan(centers).any(axis=1))
    unique, inverse = np.unique(centers[located], axis=0, return_inverse=True)
    item_lat, item_lon = np.radians(item_coords[:, 0]), np.radians(item_coords[:, 1])
    groups = []
    for g, (lat, lon) in enumerate(np.radians(unique)):
        a = (
            np.sin((item_lat - lat) / 2) ** 2
            + np.cos(lat) * np.cos(item_lat) * np.sin((item_lon - lon) / 2) ** 2
        )
        dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        groups.append((located[inverse.ravel() == g], np.flatnonzero(dist <= radius_km)))
    return groups


def search_local(
    search: GroupSearch, query_locations: np.ndarray, item_locations: np.ndarray, k: int
) -> Hits:
    return search_groups(
        search, location_groups(query_locations, item_locations), len(query_locations), k
    )


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
