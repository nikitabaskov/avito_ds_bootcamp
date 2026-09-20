"""Дообучение E5 на текстах вне выборки ранжировщика."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from candgen.core import dense, finetune
from candgen.core.data import (
    ARTIFACTS_DIR,
    SEED,
    SPLIT_DIR,
    load_corpus,
    sample_eval_queries,
    stable_hash,
)
from candgen.workflows.predict import git_state
from candgen.workflows.runs import load_corpus_embeddings

MODELS_DIR = ARTIFACTS_DIR / "models"
NEGATIVE_DEPTH = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=dense.DenseConfig.model)
    parser.add_argument("--held-out-texts", type=int, default=6000)
    parser.add_argument("--per-text", type=int, default=4)
    parser.add_argument("--negative-from", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=dense.DenseConfig.max_seq_length)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tag", default="e5ft")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    timings: dict[str, float] = {}
    rng = np.random.default_rng(SEED)
    base = dense.config_for(args.base)
    out_dir = MODELS_DIR / args.tag
    if (out_dir / "meta.json").exists():
        raise FileExistsError(f"{out_dir} already holds a trained model")

    contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
    held = sample_eval_queries(contexts, args.held_out_texts, SEED)["query_text"]
    pool = contexts.filter(~pl.col("query_text").is_in(held.implode()))
    pool = (
        pool.with_columns(order=pl.Series(stable_hash(pool["query_id"], SEED), dtype=pl.UInt64))
        .sort("order", "query_id")
        .filter(pl.int_range(pl.len()).over("query_text") < args.per_text)
        .drop("order")
        .sort("query_id")
    )
    corpus = load_corpus("split")
    item_row = {item: i for i, item in enumerate(corpus["item_id"].to_list())}
    positives = [
        np.array([item_row[i] for i in ids if i in item_row], dtype=np.int64)
        for ids in pool["item_ids"].to_list()
    ]
    text_items = (
        contexts.filter(pl.col("query_text").is_in(pool["query_text"].unique().implode()))
        .explode("item_ids", empty_as_null=False)
        .group_by("query_text")
        .agg(pl.col("item_ids").unique())
    )
    known_by_text = {
        t: np.array([item_row[i] for i in ids if i in item_row], dtype=np.int64)
        for t, ids in text_items.iter_rows()
    }
    known = [known_by_text[t] for t in pool["query_text"].to_list()]
    anchors = dense.query_texts(pool, base.query_filters)
    passages = dense.passage_texts(corpus, base.params_chars)

    t = time.perf_counter()
    model = dense.load_model(base)
    vectors = dense.encode(model, anchors, base.batch_size)
    del model
    torch.cuda.empty_cache()
    index = dense.DenseIndex(load_corpus_embeddings(base, "split", corpus, timings))
    hits, _ = index.search_local(
        vectors,
        pool["search_location_id"].to_numpy(),
        corpus["item_location_id"].to_numpy(),
        NEGATIVE_DEPTH,
    )
    del index
    torch.cuda.empty_cache()
    triples = finetune.pick_triples(
        positives, known, hits, (args.negative_from, NEGATIVE_DEPTH), rng
    )
    keys = pool["query_text"].to_numpy()[triples[0]]
    batches = [
        batch
        for _ in range(args.epochs)
        for batch in finetune.unique_batches(keys, triples[1], triples[2], args.batch_size, rng)
    ]
    timings["mining_s"] = time.perf_counter() - t
    data = {
        "held_out_texts": int(held.n_unique()),
        "train_texts": int(pool["query_text"].n_unique()),
        "contexts": pool.height,
        "triples": len(triples[0]),
        "without_negative": pool.height - len(triples[0]),
        "batches": len(batches),
    }
    print(json.dumps(data, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    model = SentenceTransformer(
        base.model, revision=base.revision, device=dense.default_device(), local_files_only=True
    )
    model.max_seq_length = args.max_seq_length
    model.gradient_checkpointing_enable()
    t = time.perf_counter()
    log = finetune.train_encoder(
        model, anchors, passages, triples, batches, args.lr, args.warmup, args.scale
    )
    timings["train_s"] = time.perf_counter() - t
    model.save(str(out_dir))
    meta = {
        "base": asdict(base),
        "args": vars(args),
        "data": data,
        "log": log,
        "git": git_state(),
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else None,
    }
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
