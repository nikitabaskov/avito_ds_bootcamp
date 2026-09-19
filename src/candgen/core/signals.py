import re

import numpy as np
import polars as pl
import torch

from candgen.core.microcats import TEMPERATURE, MicrocatIndex
from candgen.core.retrieval import Groups, order_hits, search_groups

FILTER_KEYS = (
    "Тип услуги автосервиса",
    "Предмет или специальность",
    "Кто оказывает услуги",
    "Онлайн-запись",
    "Вид услуги",
    "Тип услуги",
)
MATCH_KEYS = {
    "Вид услуги": "filt_vid",
    "Тип услуги": "filt_tip",
    "Тип услуги автосервиса": "filt_auto",
    "Предмет или специальность": "filt_subject",
}
KEY_PATTERN = re.compile("|".join(re.escape(k) for k in FILTER_KEYS))
PARAM_MARKERS = {
    "mk_online_booking": "Онлайн-запись",
    "mk_remote": "Удалённо",
    "mk_online": "Онлайн",
    "mk_city_trip": "Выезд по всему городу",
    "mk_no_trip": "Не выезжаю",
    "mk_price_list": "Название услуги",
}
MIXED_WORD = re.compile(r"[a-z][а-яё]|[а-яё][a-z]")
CORE_SHARE = 0.9


def parse_filters(text: str) -> dict[str, str]:
    found = list(KEY_PATTERN.finditer(text))
    out = {}
    ends = [m.start() for m in found[1:]] + [len(text)] if found else []
    for m, end in zip(found, ends, strict=True):
        out[m.group()] = text[m.end() : end].strip()
    return out


def filter_match_features(
    frame: pl.DataFrame, filters: list[str], params: pl.Series
) -> pl.DataFrame:
    parsed = [parse_filters(t) for t in filters]
    per_query = pl.DataFrame(
        {
            "q": np.arange(len(filters), dtype=np.int32),
            **{
                f"{name}_pat": [
                    (f"{key} {p[key]}" if key != "Тип услуги автосервиса" else p[key])
                    if p.get(key)
                    else None
                    for p in parsed
                ]
                for key, name in MATCH_KEYS.items()
            },
            "filt_booking_req": ["Онлайн-запись" in p for p in parsed],
        }
    )
    joined = (
        frame.select("q", "row")
        .join(per_query, on="q", how="left")
        .with_columns(params=params.gather(frame["row"]))
    )
    matches = {
        name: pl.when(pl.col(f"{name}_pat").is_null())
        .then(None)
        .otherwise(pl.col("params").str.contains(pl.col(f"{name}_pat"), literal=True))
        .cast(pl.Float32)
        for name in MATCH_KEYS.values()
    }
    out = joined.with_columns(**matches).with_columns(
        filt_booking=pl.when(pl.col("filt_booking_req"))
        .then(pl.col("params").str.contains("Онлайн-запись", literal=True))
        .cast(pl.Float32),
    )
    cols = [*MATCH_KEYS.values(), "filt_booking"]
    return (
        out.with_columns(
            filt_set=pl.sum_horizontal(pl.col(c).is_not_null() for c in cols).cast(pl.Float32),
            filt_hit=pl.sum_horizontal(pl.col(c).fill_null(0.0) for c in cols).cast(pl.Float32),
        )
        .with_columns(
            filt_share=pl.when(pl.col("filt_set") > 0).then(pl.col("filt_hit") / pl.col("filt_set"))
        )
        .select(*cols, "filt_set", "filt_share")
    )


def marker_features(rows: pl.Series, params: pl.Series) -> pl.DataFrame:
    text = params.gather(rows)
    return pl.DataFrame(
        {
            name: text.str.contains(marker, literal=True).cast(pl.Float32)
            for name, marker in PARAM_MARKERS.items()
        }
    )


def mixed_script_share(texts: pl.Series) -> float:
    return float(texts.str.to_lowercase().str.contains(MIXED_WORD.pattern).mean())


def text_item_means(
    pairs: pl.DataFrame, texts: pl.Series, item_row: dict[str, int], embeddings: torch.Tensor
) -> torch.Tensor:
    text_pos = {t: i for i, t in enumerate(texts.to_list())}
    unique = pairs.select("query_text", "item_id").unique()
    t_idx = torch.tensor([text_pos[t] for t in unique["query_text"]], device=embeddings.device)
    r_idx = torch.tensor([item_row[i] for i in unique["item_id"]], device=embeddings.device)
    sums = torch.zeros(len(texts), embeddings.shape[1], device=embeddings.device)
    sums.index_add_(0, t_idx, embeddings[r_idx].float())
    counts = torch.bincount(t_idx, minlength=len(texts)).clamp(min=1).unsqueeze(1)
    return sums / counts


def neighbor_centroids(
    index: MicrocatIndex, text_means: torch.Tensor, pairs: pl.DataFrame, vectors: np.ndarray
) -> np.ndarray:
    allowed = np.flatnonzero(index.texts.is_in(pairs["query_text"].unique().implode()))
    rows, sims = index.neighbors(vectors, allowed)
    weights = np.exp((sims - sims[:, :1]) / TEMPERATURE)
    weights /= weights.sum(axis=1, keepdims=True)
    w = torch.from_numpy(weights).to(text_means.device, torch.float32)
    means = text_means[torch.from_numpy(rows).to(text_means.device)]
    centroids = torch.einsum("qk,qkd->qd", w, means)
    return torch.nn.functional.normalize(centroids, dim=1).cpu().numpy()


def region_core_groups(
    pairs: pl.DataFrame,
    query_locations: np.ndarray,
    item_locations: np.ndarray,
    targets: np.ndarray,
) -> Groups:
    counts = (
        pairs.group_by("search_location_id", "item_location_id")
        .len()
        .sort(["search_location_id", "len", "item_location_id"], descending=[False, True, False])
        .with_columns(
            before=(pl.col("len").cum_sum() - pl.col("len")).over("search_location_id")
            / pl.col("len").sum().over("search_location_id")
        )
        .filter(pl.col("before") < CORE_SHARE)
    )
    cities = counts.group_by("search_location_id").agg("item_location_id")
    lookup = dict(
        zip(
            cities["search_location_id"].to_list(),
            cities["item_location_id"].to_list(),
            strict=True,
        )
    )
    groups = []
    for location in np.unique(query_locations[targets]):
        qs = np.flatnonzero(targets & (query_locations == location))
        core = lookup.get(int(location), [])
        groups.append((qs, np.flatnonzero(np.isin(item_locations, core))))
    return groups


CENTROID_FEATURES = ["nb_cos", "nb_cos_rank"]
ENCODER_FEATURES = ["enc2_sim", "enc2_rank"]
FILTER_FEATURES = ["filt_vid", "filt_tip", "filt_share"]
PAIR_BLOCK = 1_000_000


def query_centroids(
    views: list[tuple[pl.DataFrame, pl.DataFrame]],
    index: MicrocatIndex,
    query_vectors: np.ndarray,
    history_vectors: torch.Tensor,
    history_row: dict[str, int],
) -> np.ndarray:
    history = pl.concat([pairs.select("query_text", "item_id") for _, pairs in views]).unique()
    means = text_item_means(history, index.texts, history_row, history_vectors)
    centroids = np.zeros((len(query_vectors), means.shape[1]), dtype=np.float32)
    for queries, pairs in views:
        q = queries["q"].to_numpy()
        if q.size:
            centroids[q] = neighbor_centroids(index, means, pairs, query_vectors[q])
    return centroids


def pair_cosine(
    frame: pl.DataFrame, query_vectors: np.ndarray, item_vectors: torch.Tensor
) -> np.ndarray:
    q_idx, rows = frame["q"].to_numpy(), frame["row"].to_numpy()
    device = item_vectors.device
    queries = torch.from_numpy(query_vectors).to(device)
    out = np.empty(len(q_idx), dtype=np.float32)
    for start in range(0, len(q_idx), PAIR_BLOCK):
        block = slice(start, start + PAIR_BLOCK)
        a = item_vectors[torch.from_numpy(rows[block].copy()).to(device)].float()
        b = queries[torch.from_numpy(q_idx[block].copy()).to(device)].float()
        out[block] = (a * b).sum(dim=1).cpu().numpy()
    return out


def add_cosine_features(
    frame: pl.DataFrame,
    names: tuple[str, str],
    query_vectors: np.ndarray,
    item_vectors: torch.Tensor,
) -> pl.DataFrame:
    sim, rank = names
    return frame.with_columns(
        pl.Series(sim, pair_cosine(frame, query_vectors, item_vectors))
    ).with_columns(
        pl.col(sim).rank("ordinal", descending=True).over("q").cast(pl.Float32).alias(rank)
    )


def add_centroid_features(
    frame: pl.DataFrame, centroids: np.ndarray, item_vectors: torch.Tensor
) -> pl.DataFrame:
    return add_cosine_features(frame, tuple(CENTROID_FEATURES), centroids, item_vectors)


def region_hits(
    centroids: np.ndarray,
    views: list[tuple[pl.DataFrame, pl.DataFrame]],
    query_locations: np.ndarray,
    item_locations: np.ndarray,
    targets: np.ndarray,
    item_vectors: torch.Tensor,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = item_vectors.device
    cent = torch.from_numpy(centroids).to(device)

    def search(q: np.ndarray, items: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
        matrix = item_vectors[torch.from_numpy(items).to(device)].float()
        top_scores, top_rows = torch.topk(cent[torch.from_numpy(q).to(device)] @ matrix.T, depth)
        return order_hits(items[top_rows.cpu().numpy()], top_scores.cpu().numpy())

    rows = np.full((len(centroids), k), -1, dtype=np.int64)
    scores = np.full((len(centroids), k), -np.inf, dtype=np.float32)
    for queries, pairs in views:
        in_view = np.zeros(len(centroids), dtype=bool)
        in_view[queries["q"].to_numpy()] = True
        mask = in_view & targets
        if not mask.any():
            continue
        groups = region_core_groups(pairs, query_locations, item_locations, mask)
        view_rows, view_scores = search_groups(search, groups, len(centroids), k)
        rows[mask], scores[mask] = view_rows[mask], view_scores[mask]
    return rows, scores


def add_filter_features(frame: pl.DataFrame, filters: list[str], params: pl.Series) -> pl.DataFrame:
    matches = filter_match_features(frame, filters, params).select(FILTER_FEATURES)
    return frame.hstack(matches)
