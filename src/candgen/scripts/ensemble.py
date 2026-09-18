import argparse
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from catboost import CatBoostRanker

from candgen.core.data import SPLIT_DIR, load_corpus, prepare_queries
from candgen.core.diagnostics import model_ranks
from candgen.core.ensemble import RRF_K, rank_fusion, union_pool
from candgen.core.evaluation import compare_per_query
from candgen.core.features import ItemTable
from candgen.core.history import history_pairs
from candgen.core.metrics import recall_at_k
from candgen.core.ranker import make_pool
from candgen.core.submission import answer_frame, read_answer, sha256_file, validate_answer
from candgen.scripts.common import EXPERIMENTS_DIR, load_eval, peak_rss_gb
from candgen.scripts.experiment import fixed_valid_texts
from candgen.scripts.predict import (
    ITEMS_PATH,
    QUERIES_PATH,
    benchmark_history,
    expected_features,
    feature_frame,
    feature_spec,
    git_state,
    model_config,
    needs_history,
    next_output_dir,
)
from candgen.scripts.runs import RetrievalConfig, rows_to_ids

ENSEMBLES_DIR = EXPERIMENTS_DIR / "ensembles"


@dataclasses.dataclass
class Member:
    path: Path
    meta: dict
    config: RetrievalConfig
    spec: dict

    @property
    def experiment(self) -> str | None:
        try:
            return str(self.path.parent.relative_to(EXPERIMENTS_DIR))
        except ValueError:
            return None

    def frame_key(self) -> str:
        return json.dumps([dataclasses.asdict(self.config), self.spec], sort_keys=True)


def load_member(path: Path) -> Member:
    meta = json.loads(path.with_suffix(".json").read_text())
    config = model_config(meta)
    spec = feature_spec(meta)
    if meta["features"] != expected_features(spec, config):
        raise SystemExit(f"{path} was trained with a different feature set")
    return Member(path, meta, config, spec)


def member_ranks(
    members: list[Member],
    corpus_name: str,
    corpus: pl.DataFrame,
    queries: pl.DataFrame,
    run_key: str,
    pairs: pl.DataFrame | None,
    timings: dict,
) -> list[pl.DataFrame]:
    items = ItemTable(corpus)
    ranks: dict[int, pl.DataFrame] = {}
    groups: dict[str, list[int]] = {}
    for i, member in enumerate(members):
        groups.setdefault(member.frame_key(), []).append(i)
    for indices in groups.values():
        first = members[indices[0]]
        frame = feature_frame(
            first.spec, first.config, corpus_name, corpus, queries, run_key, items, pairs, timings
        )
        for i in indices:
            model = CatBoostRanker()
            model.load_model(str(members[i].path))
            scores = model.predict(make_pool(frame, members[i].meta["features"]))
            ranks[i] = model_ranks(frame, scores).select("q", "row", "model_rank")
        del frame
        torch.cuda.empty_cache()
    return [ranks[i] for i in range(len(members))]


def evaluate_dev(
    args: argparse.Namespace, members: list[Member], timings: dict, started: float
) -> dict:
    corpus, queries, _ = load_eval("dev")
    item_ids = corpus["item_id"].to_list()
    relevant = queries["item_ids"].to_list()
    pairs = None
    if any(needs_history(m.spec, m.config) for m in members):
        train_contexts = pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")
        valid_texts = fixed_valid_texts(train_contexts)
        pairs = history_pairs(
            train_contexts.filter(~pl.col("query_text").is_in(valid_texts.implode())), corpus
        )
    ranks = member_ranks(members, "split", corpus, queries, "dev", pairs, timings)
    n = queries.height

    def recall(rows: list[np.ndarray], k: int = 50) -> np.ndarray:
        return recall_at_k(rows_to_ids(item_ids, rows), relevant, k)

    fused = recall(rank_fusion(ranks, n, args.rrf_k))
    pool = recall(union_pool(ranks, n), k=np.iinfo(np.int32).max)
    baseline = args.baseline or members[0].experiment
    if baseline is None:
        raise SystemExit("--baseline is required for models outside the experiments directory")
    base = pl.read_parquet(EXPERIMENTS_DIR / baseline / "per_query.parquet").sort("q")
    if base["query_id"].to_list() != queries["query_id"].to_list():
        raise SystemExit(f"{baseline} was evaluated on different dev queries")
    per_query = base.with_columns(recall=pl.Series(fused), pool_recall=pl.Series(pool))

    member_reports = []
    for member, member_rank in zip(members, ranks, strict=True):
        own = float(recall(rank_fusion([member_rank], n, args.rrf_k)).mean())
        report_path = member.path.parent / "report.json"
        saved = json.loads(report_path.read_text())["recall@50"] if report_path.exists() else None
        member_reports.append(
            {"model": str(member.path), "recall@50": own, "report_recall@50": saved}
        )
    comparisons = {}
    references = [baseline, *(m.experiment for m in members if m.experiment is not None)]
    for reference in dict.fromkeys(references):
        path = EXPERIMENTS_DIR / reference / "per_query.parquet"
        if path.exists():
            comparisons[reference] = compare_per_query(per_query, pl.read_parquet(path))

    out_dir = ENSEMBLES_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query.write_parquet(out_dir / "per_query.parquet")
    report = {
        "ensemble": args.name,
        "mode": "dev",
        "method": f"sum 1/({args.rrf_k:g} + model_rank) over each model's full pool",
        "git": git_state(),
        "members": member_reports,
        "baseline": baseline,
        "recall@50": float(fused.mean()),
        "pool_recall": float(pool.mean()),
        "comparisons": comparisons,
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def predict_benchmark(
    args: argparse.Namespace, members: list[Member], timings: dict, started: float
) -> dict:
    queries = prepare_queries(pl.read_parquet(QUERIES_PATH))
    corpus = load_corpus("benchmark")
    item_ids = corpus["item_id"].to_list()
    query_ids = queries["query_id"].to_list()
    pairs, history_info = None, None
    if any(needs_history(m.spec, m.config) for m in members):
        pairs, history_info = benchmark_history(timings)
    ranks = member_ranks(members, "benchmark", corpus, queries, "benchmark", pairs, timings)
    predictions = rows_to_ids(item_ids, rank_fusion(ranks, queries.height, args.rrf_k))

    answer = answer_frame(query_ids, predictions)
    errors = validate_answer(answer, query_ids, item_ids)
    if errors:
        raise SystemExit("invalid answer:\n" + "\n".join(errors[:20]))
    out_dir = next_output_dir(args.tag)
    path = out_dir / "answer.csv"
    answer.write_csv(path)
    written = read_answer(path)
    if not written.equals(answer) or validate_answer(written, query_ids, item_ids):
        raise SystemExit(f"{path} does not round-trip")
    sizes = np.array([len(p) for p in predictions])
    meta = {
        "method": "ensemble",
        "rrf_k": args.rrf_k,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "members": [
            {"path": str(m.path), "sha256": sha256_file(m.path), **m.meta} for m in members
        ],
        "inference_history": history_info,
        "dev": dev_summary(args.name),
        "git": git_state(),
        "answer_sha256": sha256_file(path),
        "inputs_sha256": {p.name: sha256_file(p) for p in (QUERIES_PATH, ITEMS_PATH)},
        "queries": len(query_ids),
        "corpus_items": corpus.height,
        "answer_sizes": {"min": int(sizes.min()), "mean": float(sizes.mean())},
        "timings": {**timings, "total_s": time.perf_counter() - started},
        "peak_rss_gb": peak_rss_gb(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"saved {path}")
    return meta


def dev_summary(name: str) -> dict | None:
    path = ENSEMBLES_DIR / name / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return {k: report[k] for k in ("recall@50", "pool_recall", "members")} | {"report": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=["dev", "benchmark"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--rrf-k", type=float, default=RRF_K)
    parser.add_argument("--tag", default="ensemble")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("duplicate models")
    members = [load_member(p) for p in args.models]
    started = time.perf_counter()
    timings: dict[str, float] = {}
    run = evaluate_dev if args.mode == "dev" else predict_benchmark
    report = run(args, members, timings, started)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
