import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from candgen.core.evaluation import paired_bootstrap
from candgen.core.metrics import recall_at_k
from candgen.core.ranker import POINTWISE, RankerConfig, fit_model, make_pool, select_top
from candgen.scripts.common import EXPERIMENTS_DIR, peak_rss_gb
from candgen.scripts.experiment import build_parser, configure, prepare, train_frames
from candgen.scripts.predict import git_state
from candgen.scripts.runs import rows_to_ids

REFERENCES = [f"EXP-009/pool400x300_s{seed}" for seed in (42, 43, 44)]
GPU_MAX_GROUP = 1023


def reference_recall(names: list[str], query_ids: list[str]) -> pl.DataFrame:
    frames = [
        pl.read_parquet(EXPERIMENTS_DIR / n / "per_query.parquet").select(
            "query_id", "has_center", "filters_set", pl.col("recall").alias(n)
        )
        for n in names
    ]
    joined = frames[0]
    for f in frames[1:]:
        joined = joined.join(f.drop("has_center", "filters_set"), on="query_id")
    order = pl.DataFrame({"query_id": query_ids})
    return order.join(joined, on="query_id", how="left", maintain_order="left").with_columns(
        ref=pl.mean_horizontal(names)
    )


def compare(recall: np.ndarray, ref: pl.DataFrame) -> dict:
    out = {"recall@50": float(recall.mean()), **paired_bootstrap(recall, ref["ref"].to_numpy())}
    for name, mask in {
        "no_center": ~ref["has_center"],
        "filters_set": ref["filters_set"],
    }.items():
        m = mask.to_numpy()
        out[name] = {
            "n": int(m.sum()),
            "recall@50": float(recall[m].mean()),
            **paired_bootstrap(recall[m], ref["ref"].to_numpy()[m]),
        }
    return out


def to_probability(scores: np.ndarray, loss: str) -> np.ndarray:
    return 1 / (1 + np.exp(-scores)) if loss in POINTWISE else scores


def main() -> None:
    parser = build_parser(required=False)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--lab", required=True)
    parser.add_argument("--references", nargs="+", default=REFERENCES)
    args = parser.parse_args()
    retrieval, _, features = configure(parser, args)
    variants = json.loads(args.variants.read_text())
    out_dir = EXPERIMENTS_DIR / "lab" / args.lab
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    started = time.perf_counter()
    timings: dict[str, float] = {}
    setup = prepare(args, retrieval, timings)
    frames = train_frames(
        args,
        retrieval,
        setup["corpus"],
        setup["train_contexts"],
        setup["pairs"],
        setup["items"],
        setup["item_vectors"],
        setup["scorer"],
        setup["index"],
        timings,
    )
    dev_frame, dev_queries = setup["dev_frame"], setup["dev_queries"]
    item_ids = setup["item_ids"]
    del setup
    torch.cuda.empty_cache()
    dev_pool = make_pool(dev_frame, features)
    pools = {}

    def train_pools(cap: int | None) -> tuple:
        if cap not in pools:
            fit, valid = frames["fit"], frames["valid"]
            if cap:
                fit = fit.filter(pl.int_range(pl.len()).over("q") < cap)
                valid = valid.filter(pl.int_range(pl.len()).over("q") < cap)
            pools[cap] = make_pool(fit, features), make_pool(valid, features)
        return pools[cap]

    relevant = dev_queries["item_ids"].to_list()
    ref = reference_recall(args.references, dev_queries["query_id"].to_list())
    timings["setup_s"] = time.perf_counter() - started

    def dev_recall(scores: np.ndarray) -> np.ndarray:
        rows = select_top(dev_frame, scores, dev_queries.height)
        return recall_at_k(rows_to_ids(item_ids, rows), relevant, 50)

    for variant in variants:
        name = variant["name"]
        if name in results:
            print(f"{name}: done, skipped")
            continue
        config = RankerConfig(
            loss_function=variant.get("loss", "YetiRank"),
            iterations=variant.get("iterations", 500),
            learning_rate=variant.get("lr", 0.05),
            depth=variant.get("depth", 6),
            early_stopping_rounds=0,
            task_type=variant.get("task_type", "GPU"),
        )
        seeds = variant.get("seeds", [42])
        capped = config.task_type == "GPU" and config.loss_function not in POINTWISE
        fit_pool, valid_pool = train_pools(GPU_MAX_GROUP if capped else None)
        runs, probabilities = [], []
        for seed in seeds:
            t = time.perf_counter()
            model = fit_model(
                fit_pool,
                valid_pool,
                dataclasses.replace(config, random_seed=seed),
                {"verbose": 0, "metric_period": 50, **variant.get("params", {})},
            )
            train_s = time.perf_counter() - t
            scores = model.predict(dev_pool)
            recall = dev_recall(scores)
            probabilities.append(to_probability(scores, config.loss_function))
            runs.append({"seed": seed, "recall@50": float(recall.mean()), "train_s": train_s})
            print(f"{name} seed {seed}: {recall.mean():.5f} in {train_s:.0f}s", flush=True)
            runs[-1]["_recall"] = recall
        seed_mean = np.mean([r.pop("_recall") for r in runs], axis=0)
        entry = {
            "variant": variant,
            "config": dataclasses.asdict(config),
            "train_group_cap": GPU_MAX_GROUP if capped else None,
            "runs": runs,
            "seed_mean": compare(seed_mean, ref),
        }
        if len(seeds) > 1:
            entry["ensemble"] = compare(dev_recall(np.mean(probabilities, axis=0)), ref)
        results[name] = entry
        summary = entry.get("ensemble", entry["seed_mean"])
        print(
            f"{name}: seed mean {entry['seed_mean']['recall@50']:.5f}"
            f" ({entry['seed_mean']['diff'] * 100:+.2f} pp), "
            f"{'ensemble' if len(seeds) > 1 else 'single'} {summary['recall@50']:.5f}"
            f" [{summary['ci95'][0] * 100:+.2f}; {summary['ci95'][1] * 100:+.2f}]",
            flush=True,
        )
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    meta = {
        "git": git_state(),
        "features": features,
        "references": args.references,
        "reference_recall@50": float(ref["ref"].mean()),
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
