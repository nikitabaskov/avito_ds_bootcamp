import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from candgen.core import dense
from candgen.core.evaluation import paired_bootstrap
from candgen.scripts.common import EXPERIMENTS_DIR, peak_rss_gb
from candgen.scripts.predict import git_state
from candgen.scripts.runs import RUNS_DIR, dense_runs, load_corpus_embeddings, query_vectors
from candgen.scripts.signal_diagnostics import MODEL, band_auc, recall_with, scored_dev_frame

LIST_KS = (50, 100, 200, 400)
EXTRA_KS = {"global": (100, 200, 400), "local": (100, 300)}
CONTROL_K = 800


def list_recall(hits: np.ndarray, k: int, relevant: list[set[int]]) -> np.ndarray:
    return np.array(
        [len(rel & set(row[:k].tolist())) / len(rel) for row, rel in zip(hits, relevant)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    timings: dict[str, float] = {}
    dev = scored_dev_frame(args.model, timings)
    frame, config, corpus, queries = dev["frame"], dev["config"], dev["corpus"], dev["queries"]
    relevant = dev["relevant"]
    base = config.dense_config
    second = dense.config_for(
        args.encoder, params_chars=base.params_chars, query_filters=base.query_filters
    )
    run_dir = RUNS_DIR / "dev"

    t = time.perf_counter()
    base_runs = dense_runs(base, "split", corpus, queries, run_dir, timings)
    new_runs = dense_runs(second, "split", corpus, queries, run_dir, timings)
    timings["lists_s"] = time.perf_counter() - t
    lists = {}
    for scope in ("global", "local"):
        for k in LIST_KS:
            old = list_recall(base_runs[f"dense_{scope}"][0], k, relevant)
            new = list_recall(new_runs[f"dense_{scope}"][0], k, relevant)
            lists[f"{scope}_{k}"] = {
                "base": float(old.mean()),
                "encoder": float(new.mean()),
                **paired_bootstrap(new, old),
            }

    pool = [set() for _ in range(queries.height)]
    for q, row in zip(frame["q"].to_list(), frame["row"].to_list(), strict=True):
        pool[q].add(row)
    pool_base = recall_with(pool, None, relevant)
    pool_ext = {}
    for scope, ks in EXTRA_KS.items():
        for k in ks:
            extra = new_runs[f"dense_{scope}"][0][:, :k]
            rec = recall_with(pool, extra, relevant)
            added = np.mean([len(set(row[row >= 0].tolist()) - p) for row, p in zip(extra, pool)])
            pool_ext[f"{scope}_{k}"] = {
                "pool_recall": float(rec.mean()),
                "added_per_query": float(added),
                **paired_bootstrap(rec, pool_base),
            }
    control = base_runs["dense_global"][0][:, :CONTROL_K]
    rec = recall_with(pool, control, relevant)
    pool_ext[f"control_base_global_{CONTROL_K}"] = {
        "pool_recall": float(rec.mean()),
        "added_per_query": float(
            np.mean([len(set(row[row >= 0].tolist()) - p) for row, p in zip(control, pool)])
        ),
        **paired_bootstrap(rec, pool_base),
    }

    t = time.perf_counter()
    embeddings = load_corpus_embeddings(second, "split", corpus, timings)
    vectors = query_vectors(second, queries, run_dir, timings)
    device = dense.default_device()
    item_vectors = torch.from_numpy(embeddings).to(device)
    query_matrix = torch.from_numpy(vectors).to(device)
    q_idx, rows = frame["q"].to_numpy(), frame["row"].to_numpy()
    sim = np.empty(len(q_idx), dtype=np.float32)
    for start in range(0, len(q_idx), 1_000_000):
        block = slice(start, start + 1_000_000)
        a = item_vectors[torch.from_numpy(rows[block].copy()).to(device)]
        b = query_matrix[torch.from_numpy(q_idx[block].copy()).to(device)]
        sim[block] = (a * b).sum(dim=1).cpu().numpy()
    frame = frame.with_columns(enc_sim=pl.Series(sim)).with_columns(
        enc_rank=pl.col("enc_sim").rank("ordinal", descending=True).over("q"),
        dense_rank=pl.col("dense_sim").rank("ordinal", descending=True).over("q"),
    )
    timings["similarity_s"] = time.perf_counter() - t

    lost = frame.filter((pl.col("label") == 1) & (pl.col("model_rank") > 50))
    in_band = frame.filter(pl.col("model_rank") <= 200)
    report = {
        "encoder": args.encoder,
        "encoder_config": {"revision": second.revision, "tag": second.passage_tag()},
        "model": str(args.model),
        "git": git_state(),
        "queries": queries.height,
        "list_recall": lists,
        "pool_recall": float(pool_base.mean()),
        "pool_extension": pool_ext,
        "band_auc": {name: band_auc(frame, name) for name in ("enc_sim", "dense_sim")},
        "enc_sim_spearman_dense_sim_in_band": float(
            in_band.select(pl.corr("enc_sim", "dense_sim", method="spearman")).item()
        ),
        "lost_positives": lost.height,
        "lost_in_pool_top50": {
            "enc_sim": float((lost["enc_rank"] <= 50).mean()),
            "dense_sim": float((lost["dense_rank"] <= 50).mean()),
        },
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    output = args.output or EXPERIMENTS_DIR / "stage1" / "encoders" / second.passage_tag()
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (output / "report.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
