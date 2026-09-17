import json
import resource

import polars as pl

from candgen.core.data import ARTIFACTS_DIR, SPLIT_DIR, load_corpus

EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"


def load_eval(part: str) -> tuple[pl.DataFrame, pl.DataFrame, set[str]]:
    corpus = load_corpus("split")
    queries = pl.read_parquet(SPLIT_DIR / f"eval_{part}.parquet")
    seen_items = set(
        pl.read_parquet(SPLIT_DIR / "contexts_train.parquet")["item_ids"].explode().unique()
    )
    return corpus, queries, seen_items


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def write_report(name: str, report: dict) -> None:
    out = EXPERIMENTS_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    out.write_text(text)
    print(text)
    print(f"saved {out}")
