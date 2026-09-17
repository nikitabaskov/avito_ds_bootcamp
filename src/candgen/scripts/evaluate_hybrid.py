import argparse
import dataclasses

import torch

from candgen.core import bm25, dense
from candgen.core.evaluation import recall_report
from candgen.scripts.common import load_eval, peak_rss_gb, write_report
from candgen.scripts.runs import (
    CHANNELS,
    GLOBAL_DEPTH,
    LOCAL_DEPTH,
    RetrievalConfig,
    fuse,
    retrieve,
    rows_to_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["dev", "test"], default="dev")
    parser.add_argument("--channels", nargs="+", choices=CHANNELS, default=list(CHANNELS))
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-k", type=int, default=300)
    parser.add_argument("--local-k", type=int, default=200)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--title-repeat", type=int, default=3)
    parser.add_argument("--bm25-query-filters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dense-query-filters", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--params-chars", type=int, default=dense.DenseConfig.params_chars)
    args = parser.parse_args()
    if args.global_k > GLOBAL_DEPTH or args.local_k > LOCAL_DEPTH:
        parser.error(f"global-k <= {GLOBAL_DEPTH}, local-k <= {LOCAL_DEPTH}")
    config = RetrievalConfig(
        channels=tuple(c for c in CHANNELS if c in args.channels),
        local=args.local,
        global_k=args.global_k,
        local_k=args.local_k,
        rrf_k=args.rrf_k,
        bm25_config=bm25.BM25Config(
            title_repeat=args.title_repeat, query_filters=args.bm25_query_filters
        ),
        dense_config=dense.DenseConfig(
            params_chars=args.params_chars, query_filters=args.dense_query_filters
        ),
    )

    corpus, queries, seen_items = load_eval(args.part)
    item_ids = corpus["item_id"].to_list()

    timings: dict[str, float] = {}
    runs = retrieve(config, "split", corpus, queries, args.part, timings)

    channel_reports = {}
    for name, (rows, _) in runs.items():
        full = recall_report(queries, rows_to_ids(item_ids, rows), seen_items)
        channel_reports[name] = {
            "depth": int(rows.shape[1]),
            **{k: full[k] for k in ("recall", "mean_pool_size", "empty_pool_queries")},
        }

    fused = fuse(config, runs, timings)
    report = {
        "method": "rrf",
        "part": args.part,
        "config": {
            "channels": list(config.channels),
            "local": config.local,
            "global_k": config.global_k,
            "local_k": config.local_k,
            "rrf_k": config.rrf_k,
            "bm25": dataclasses.asdict(config.bm25_config) if "bm25" in config.channels else None,
            "dense": dataclasses.asdict(config.dense_config)
            if "dense" in config.channels
            else None,
        },
        "corpus_items": corpus.height,
        **recall_report(queries, rows_to_ids(item_ids, fused), seen_items),
        "channels": channel_reports,
        "timings": timings,
        "peak_rss_gb": peak_rss_gb(),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else None,
    }
    write_report(config.report_name(args.part), report)


if __name__ == "__main__":
    main()
