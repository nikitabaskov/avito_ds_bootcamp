"""Обучение CatBoost и стабильный top-k с RRF как вторичным порядком."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from catboost import CatBoost, CatBoostRanker, Pool

from candgen.core.features import FEATURES
from candgen.core.metrics import recall_at_k

POINTWISE = ("Logloss", "CrossEntropy")


@dataclass(frozen=True)
class RankerConfig:
    loss_function: str = "YetiRank"
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    early_stopping_rounds: int = 200
    random_seed: int = 42
    task_type: str = "CPU"


def make_pool(frame: pl.DataFrame, features: Sequence[str] = FEATURES) -> Pool:
    return Pool(
        data=frame.select(features).to_numpy().astype(np.float32),
        label=frame["label"].to_numpy() if "label" in frame.columns else None,
        group_id=frame["q"].to_numpy(),
        feature_names=list(features),
    )


def train_ranker(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    config: RankerConfig,
    features: Sequence[str] = FEATURES,
    params: dict | None = None,
) -> CatBoost:
    return fit_model(make_pool(train, features), make_pool(valid, features), config, params)


def fit_model(
    train: Pool, valid: Pool, config: RankerConfig, params: dict | None = None
) -> CatBoost:
    options = {
        "loss_function": config.loss_function,
        "iterations": config.iterations,
        "learning_rate": config.learning_rate,
        "depth": config.depth,
        "random_seed": config.random_seed,
        "task_type": config.task_type,
        "eval_metric": "RecallAt:top=50",
        "early_stopping_rounds": config.early_stopping_rounds or None,
        "use_best_model": config.early_stopping_rounds > 0,
        "allow_writing_files": False,
        "verbose": 100,
        **(params or {}),
    }
    model = CatBoost(options) if config.loss_function in POINTWISE else CatBoostRanker(**options)
    model.fit(train, eval_set=valid)
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


def full_recall_curve(
    model: CatBoostRanker,
    frame: pl.DataFrame,
    queries: pl.DataFrame,
    item_ids: Sequence[str],
    features: Sequence[str],
    points: Sequence[int],
    k: int = 50,
) -> dict[int, float]:
    """Evaluate all held-out queries against their original positives, including pool misses.

    queries.q keeps the original (possibly sparse) group IDs used by frame.
    """
    pool = make_pool(frame, features)
    relevant = queries["item_ids"].to_list()
    group_ids = queries["q"].to_list()
    curve = {}
    for n in points:
        rows = select_top(frame, model.predict(pool, ntree_end=n), max(group_ids) + 1, k)
        predicted = [[item_ids[r] for r in rows[q]] for q in group_ids]
        curve[n] = float(recall_at_k(predicted, relevant, k).mean())
    return curve
