from collections.abc import Collection, Sequence

import numpy as np


def recall_at_k(
    predictions: Sequence[Sequence[str]], relevant: Sequence[Collection[str]], k: int
) -> np.ndarray:
    if len(predictions) != len(relevant):
        raise ValueError(f"{len(predictions)} predictions for {len(relevant)} queries")
    out = np.empty(len(relevant), dtype=np.float64)
    for i, (pred, rel) in enumerate(zip(predictions, relevant, strict=True)):
        rel = set(rel)
        if not rel:
            raise ValueError(f"query {i} has no relevant items")
        out[i] = len(set(pred[:k]) & rel) / len(rel)
    return out
