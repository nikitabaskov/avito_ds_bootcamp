import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from catboost import CatBoostRanker

from candgen.core import signals
from candgen.core.data import SPLIT_DIR, load_corpus
from candgen.core.dense import DenseIndex, default_device
from candgen.core.evaluation import paired_bootstrap
from candgen.core.features import ItemTable
from candgen.core.history import history_pairs
from candgen.core.ranker import make_pool
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.experiment import fixed_valid_texts
from candgen.scripts.predict import feature_frame, feature_spec, git_state, model_config
from candgen.scripts.runs import (
    RUNS_DIR,
    load_corpus_embeddings,
    microcat_index,
    query_vectors,
    text_vectors,
)

MODEL = EXPERIMENTS_DIR / "EXP-009" / "pool400x300_s42" / "model.cbm"
BAND = 200
LIST_KS = (50, 100, 200)
REFERENCE = ["dense_sim", "mc_prob", "title_overlap", "score_bm25_global"]


def band_auc(frame: pl.DataFrame, feature: str, query_mask: pl.Expr | None = None) -> dict:
    band = frame.filter(pl.col("model_rank") <= BAND)
    if query_mask is not None:
        band = band.filter(query_mask)
    scored = band.with_columns(f=pl.col(feature).fill_null(-1e9).fill_nan(-1e9)).with_columns(
        rank_all=pl.col("f").rank("average").over("q"),
        rank_group=pl.col("f").rank("average").over("q", "label"),
        n_neg=(pl.col("label") == 0).sum().over("q"),
    )
    pos = scored.filter((pl.col("label") == 1) & (pl.col("n_neg") > 0)).with_columns(
        auc=(pl.col("rank_all") - pl.col("rank_group")) / pl.col("n_neg")
    )
    lost = pos.filter(pl.col("model_rank") > 50)
    return {
        "positives": pos.height,
        "auc_band": float(pos["auc"].mean()) if pos.height else None,
        "lost_positives": lost.height,
        "auc_lost": float(lost["auc"].mean()) if lost.height else None,
    }


def recall_with(
    pool: list[set[int]], extra: np.ndarray | None, relevant: list[set[int]]
) -> np.ndarray:
    out = np.empty(len(relevant))
    for q, rel in enumerate(relevant):
        found = pool[q] if extra is None else pool[q] | set(extra[q][extra[q] >= 0].tolist())
        out[q] = len(rel & found) / len(rel)
    return out


def scored_dev_frame(model_path: Path, timings: dict) -> dict:
    meta = json.loads(model_path.with_suffix(".json").read_text())
    config, spec = model_config(meta), feature_spec(meta)
    corpus, queries, _ = load_eval("dev")
    contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
    valid_texts = fixed_valid_texts(contexts)
    pairs = history_pairs(
        contexts.filter(~pl.col("query_text").is_in(valid_texts.implode())), corpus
    )
    items = ItemTable(corpus)
    item_row = {item: i for i, item in enumerate(corpus["item_id"].to_list())}
    relevant = [{item_row[i] for i in ids} for ids in queries["item_ids"].to_list()]

    frame = feature_frame(spec, config, "split", corpus, queries, "dev", items, pairs, timings)
    model = CatBoostRanker()
    model.load_model(str(model_path))
    labels = pl.DataFrame(
        [(q, r) for q, rel in enumerate(relevant) for r in rel], schema=["q", "row"], orient="row"
    ).with_columns(
        pl.col("q").cast(pl.Int32), pl.col("row").cast(pl.Int64), label=pl.lit(1, pl.Int8)
    )
    frame = (
        frame.with_columns(score=pl.Series(model.predict(make_pool(frame, meta["features"]))))
        .sort(["q", "score", "rrf_rank"], descending=[False, True, False])
        .with_columns(model_rank=pl.int_range(1, pl.len() + 1).over("q"))
        .join(labels, on=["q", "row"], how="left")
        .with_columns(pl.col("label").fill_null(0))
    )
    return {
        "frame": frame,
        "config": config,
        "corpus": corpus,
        "queries": queries,
        "pairs": pairs,
        "items": items,
        "item_row": item_row,
        "relevant": relevant,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_DIR / "stage1" / "signals")
    args = parser.parse_args()

    started = time.perf_counter()
    timings: dict[str, float] = {}
    dev = scored_dev_frame(args.model, timings)
    frame, config, corpus, queries = dev["frame"], dev["config"], dev["corpus"], dev["queries"]
    pairs, items, item_row, relevant = dev["pairs"], dev["items"], dev["item_row"], dev["relevant"]

    t = time.perf_counter()
    embeddings = load_corpus_embeddings(config.dense_config, "split", corpus, timings)
    device = default_device()
    item_vectors = torch.from_numpy(embeddings).to(device)
    index = microcat_index(config.dense_config, pairs, timings)
    means = signals.text_item_means(pairs, index.texts, item_row, item_vectors)
    vectors = text_vectors(
        config.dense_config,
        queries["query_text"].to_list(),
        RUNS_DIR / "dev" / "e5_texts.npz",
        timings,
    )
    centroids = signals.neighbor_centroids(index, means, pairs, vectors)
    q_idx, rows = frame["q"].to_numpy(), frame["row"].to_numpy()
    cent = torch.from_numpy(centroids).to(device)
    nb_cos = np.empty(len(q_idx), dtype=np.float32)
    for start in range(0, len(q_idx), 1_000_000):
        sl = slice(start, start + 1_000_000)
        a = item_vectors[torch.from_numpy(rows[sl]).to(device)].float()
        b = cent[torch.from_numpy(q_idx[sl]).to(device)]
        nb_cos[sl] = (a * b).sum(dim=1).cpu().numpy()
    params = corpus["item_infm_params_text"].fill_null("")
    frame = pl.concat(
        [
            frame.with_columns(nb_cos=pl.Series(nb_cos)),
            signals.filter_match_features(
                frame, queries["search_infm_params_text"].to_list(), params
            ),
            signals.marker_features(frame["row"], params),
        ],
        how="horizontal_extend",
    ).with_columns(
        nb_cos_rank=pl.col("nb_cos").rank("ordinal", descending=True).over("q"),
    )
    timings["signals_s"] = time.perf_counter() - t

    lost = frame.filter((pl.col("label") == 1) & (pl.col("model_rank") > 50))
    has_filters = pl.col("filt_set") > 0
    features = {
        "centroid": ["nb_cos"],
        "filters": [
            "filt_share",
            "filt_vid",
            "filt_tip",
            "filt_auto",
            "filt_subject",
            "filt_booking",
        ],
        "markers": list(signals.PARAM_MARKERS),
    }
    auc = {
        name: band_auc(frame, name)
        for name in [*REFERENCE, *features["centroid"], *features["markers"]]
    }
    auc |= {
        f"{name}|filters_set": band_auc(frame, name, has_filters)
        for name in [*REFERENCE, *features["filters"]]
    }
    in_band = frame.filter(pl.col("model_rank") <= BAND)
    correlation = {
        ref: float(in_band.select(pl.corr("nb_cos", ref, method="spearman")).item())
        for ref in REFERENCE
    }
    positives = frame.filter(pl.col("label") == 1)
    pool_all = frame.filter(has_filters)
    filter_rates = {
        name: {
            "positives": float(positives.filter(pl.col(name).is_not_null())[name].mean() or 0.0),
            "pool": float(pool_all.filter(pl.col(name).is_not_null())[name].mean() or 0.0),
            "lost_positives": float(lost.filter(pl.col(name).is_not_null())[name].mean() or 0.0),
            "n_queries": int(frame.filter(pl.col(name).is_not_null())["q"].n_unique()),
        }
        for name in features["filters"]
    }
    marker_rates = {
        name: {
            "positives": float(positives[name].mean()),
            "pool": float(frame[name].mean()),
            "model_top50": float(frame.filter(pl.col("model_rank") <= 50)[name].mean()),
            "lost_positives": float(lost[name].mean()),
        }
        for name in features["markers"]
    }
    lost_nb_top50 = float((lost["nb_cos_rank"] <= 50).mean())

    pool = [set() for _ in range(queries.height)]
    for q, row in zip(q_idx.tolist(), rows.tolist(), strict=True):
        pool[q].add(row)
    base = recall_with(pool, None, relevant)
    t = time.perf_counter()
    dense = DenseIndex(embeddings, device)
    centroid_lists = {}
    for k in LIST_KS:
        hits, _ = dense.search(centroids, k)
        rec = recall_with(pool, hits, relevant)
        centroid_lists[f"global_{k}"] = {
            "pool_recall": float(rec.mean()),
            **paired_bootstrap(rec, base),
        }
    item_locations = corpus["item_location_id"].to_numpy()
    query_locations = queries["search_location_id"].to_numpy()
    for k in (50, 100):
        hits, _ = dense.search_local(centroids, query_locations, item_locations, k)
        rec = recall_with(pool, hits, relevant)
        centroid_lists[f"local_{k}"] = {
            "pool_recall": float(rec.mean()),
            **paired_bootstrap(rec, base),
        }

    no_center = ~np.isin(
        query_locations,
        items.locations.filter(pl.col("location_items") > 0)["item_location_id"].to_numpy(),
    )
    groups = signals.region_core_groups(pairs, query_locations, item_locations, no_center)
    core_items = {int(q): set(g.tolist()) for qs, g in groups for q in qs}
    covered = [
        np.mean([r in core_items[q] for r in relevant[q]]) for q in np.flatnonzero(no_center)
    ]
    qvec = query_vectors(config.dense_config, queries, RUNS_DIR / "dev", timings)
    region = {
        "queries": int(no_center.sum()),
        "positives_in_core_cities": float(np.mean(covered)),
        "core_cities_items_median": float(np.median([len(g) for _, g in groups])),
        "pool_recall": float(base[no_center].mean()),
    }
    for name, vecs in (("query", qvec), ("centroid", centroids)):
        for k in (100, 200):
            hits, _ = dense.search_groups(vecs, groups, k)
            rec = recall_with(pool, hits, relevant)
            region[f"{name}_{k}"] = {
                "pool_recall": float(rec[no_center].mean()),
                **paired_bootstrap(rec[no_center], base[no_center]),
            }
    timings["lists_s"] = time.perf_counter() - t

    bench_queries = pl.read_parquet("data/input/benchmark_queries.parquet")
    mixed = {
        "dev_queries": signals.mixed_script_share(queries["query_text"]),
        "benchmark_queries": signals.mixed_script_share(bench_queries["search_query"]),
        "benchmark_titles": signals.mixed_script_share(
            load_corpus("benchmark")["item_title_raw"].fill_null("")
        ),
        "examples": bench_queries["search_query"]
        .filter(
            bench_queries["search_query"]
            .str.to_lowercase()
            .str.contains(signals.MIXED_WORD.pattern)
        )
        .head(15)
        .to_list(),
    }

    report = {
        "model": str(args.model),
        "git": git_state(),
        "queries": queries.height,
        "pool_recall": float(base.mean()),
        "lost_positives": lost.height,
        "band_auc": auc,
        "nb_cos_spearman_in_band": correlation,
        "lost_positives_in_nb_cos_top50": lost_nb_top50,
        "filter_match_rates": filter_rates,
        "marker_rates": marker_rates,
        "centroid_lists": centroid_lists,
        "region_core": region,
        "mixed_script": mixed,
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (args.output / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
