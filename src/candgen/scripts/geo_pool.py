import argparse
import dataclasses
import json
import time

import numpy as np
import polars as pl

from candgen.core.data import SPLIT_DIR
from candgen.core.evaluation import paired_bootstrap
from candgen.core.features import ItemTable
from candgen.core.history import full_view, history_pairs, query_centers
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.experiment import fixed_valid_texts
from candgen.scripts.pool_recall import evaluate
from candgen.scripts.predict import git_state
from candgen.scripts.runs import RetrievalConfig

BASE = RetrievalConfig(global_k=400, local_k=300, radius_km=25, radius_k=50)
PENALTIES = ((1.0, 0.0), (0.8, 0.01), (0.6, 0.02), (0.6, 0.04))
LOCAL_PENALTIES = ((1.0, 0.0), (0.8, 0.01), (0.6, 0.02), (0.4, 0.04), (0.2, 0.08))


def add_grid() -> dict[str, RetrievalConfig]:
    variants = {}
    for km in (50, 100):
        for k in (50, 100):
            for weight, delta in PENALTIES:
                variants[f"geo{km}x{k}_w{weight:g}_d{delta:g}"] = dataclasses.replace(
                    BASE,
                    radius_k=0,
                    radius_km=0.0,
                    geo_km=km,
                    geo_k=k,
                    geo_weight=weight,
                    geo_delta=delta,
                )
    variants["base+geo100x50_w0.6_d0.02"] = dataclasses.replace(
        BASE, geo_km=100, geo_k=50, geo_weight=0.6, geo_delta=0.02
    )
    return variants


def local_grid() -> dict[str, RetrievalConfig]:
    variants = {f"base_l{k}": dataclasses.replace(BASE, local_k=k) for k in (400, 500)}
    for km in (50, 100):
        for k in (350, 450):
            for weight, delta in LOCAL_PENALTIES:
                variants[f"nolocal_geo{km}x{k}_w{weight:g}_d{delta:g}"] = dataclasses.replace(
                    BASE,
                    local_k=0,
                    radius_k=0,
                    radius_km=0.0,
                    geo_km=km,
                    geo_k=k,
                    geo_weight=weight,
                    geo_delta=delta,
                )
    return variants


GRIDS = {"add": ("geo", add_grid), "replace_local": ("geo_local", local_grid)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=list(GRIDS), default="add")
    args = parser.parse_args()
    folder, grid = GRIDS[args.grid]
    started = time.perf_counter()
    timings: dict[str, float] = {}
    corpus, queries, _ = load_eval("dev")
    contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
    valid_texts = fixed_valid_texts(contexts)
    pairs = history_pairs(
        contexts.filter(~pl.col("query_text").is_in(valid_texts.implode())), corpus
    )
    centers = query_centers(full_view(queries, pairs), ItemTable(corpus))

    variants = {"base": BASE, **grid()}

    results = {
        name: evaluate(config, corpus, queries, centers, timings)
        for name, config in variants.items()
    }
    base = results["base"]
    other = (
        queries.select(
            pl.col("search_location_id"),
            pl.col("item_ids"),
        )
        .with_row_index("q")
        .explode("item_ids")
        .join(
            corpus.select(pl.col("item_id").alias("item_ids"), "item_location_id"),
            on="item_ids",
        )
        .group_by("q")
        .agg((pl.col("item_location_id") != pl.col("search_location_id")).any().alias("other"))
        .sort("q")
    )
    other_mask = np.zeros(queries.height, dtype=bool)
    other_mask[other.filter("other")["q"].to_numpy()] = True
    report = {
        "git": git_state(),
        "queries": queries.height,
        "queries_with_center": int((~np.isnan(centers).any(axis=1)).sum()),
        "queries_with_other_location_positive": int(other_mask.sum()),
        "variants": {
            name: {
                "config": dataclasses.asdict(variants[name]) | {"bm25_config": None},
                "pool_recall": float(r["pool"].mean()),
                "pool_size": float(r["pool_size"].mean()),
                "vs_base": paired_bootstrap(r["pool"], base["pool"]),
                "other_location_queries": paired_bootstrap(
                    r["pool"][other_mask], base["pool"][other_mask]
                ),
            }
            for name, r in results.items()
        },
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    out = EXPERIMENTS_DIR / "stage1" / folder / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    for name, v in report["variants"].items():
        print(
            f"{name:36s} pool {v['pool_recall']:.5f} size {v['pool_size']:7.1f} "
            f"diff {v['vs_base']['diff'] * 100:+.2f} [{v['vs_base']['ci95'][0] * 100:+.2f}; "
            f"{v['vs_base']['ci95'][1] * 100:+.2f}]"
        )


if __name__ == "__main__":
    main()
