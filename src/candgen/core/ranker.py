from dataclasses import dataclass

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from candgen.core.features import FEATURES


@dataclass(frozen=True)
class RankerConfig:
    loss_function: str = "YetiRank"
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    early_stopping_rounds: int = 200
    random_seed: int = 42
    task_type: str = "CPU"


def make_pool(frame: pl.DataFrame) -> Pool:
    return Pool(
        data=frame.select(FEATURES).to_numpy().astype(np.float32),
        label=frame["label"].to_numpy() if "label" in frame.columns else None,
        group_id=frame["q"].to_numpy(),
        feature_names=FEATURES,
    )


def train_ranker(train: pl.DataFrame, valid: pl.DataFrame, config: RankerConfig) -> CatBoostRanker:
    model = CatBoostRanker(
        loss_function=config.loss_function,
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        random_seed=config.random_seed,
        task_type=config.task_type,
        eval_metric="RecallAt:top=50",
        early_stopping_rounds=config.early_stopping_rounds,
        use_best_model=True,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(make_pool(train), eval_set=make_pool(valid))
    return model


def select_top(
    frame: pl.DataFrame, scores: np.ndarray, n_queries: int, k: int = 50
) -> list[np.ndarray]:
    top = (
        frame.select("q", "row", "rrf_rank")
        .with_columns(score=pl.Series(scores, dtype=pl.Float64))
        .sort(["q", "score", "rrf_rank"], descending=[False, True, False])
        .group_by("q", maintain_order=True)
        .head(k)
        .group_by("q", maintain_order=True)
        .agg("row")
    )
    out = [np.empty(0, dtype=np.int64) for _ in range(n_queries)]
    for q, rows in top.iter_rows():
        out[q] = np.array(rows, dtype=np.int64)
    return out
