import hashlib
import re
from collections.abc import Collection, Sequence
from pathlib import Path

import polars as pl

ANSWER_K = 50
ID_LENGTH = 16
ITEM_ID_RE = re.compile(r"[0-9a-f]{16}")


def answer_frame(query_ids: Sequence[str], predictions: Sequence[Sequence[str]]) -> pl.DataFrame:
    if len(query_ids) != len(predictions):
        raise ValueError(f"{len(predictions)} predictions for {len(query_ids)} queries")
    return pl.DataFrame(
        {"query_id": list(query_ids), "answer": [" ".join(p[:ANSWER_K]) for p in predictions]},
        schema={"query_id": pl.String, "answer": pl.String},
    )


def validate_answer(
    answer: pl.DataFrame, query_ids: Collection[str], item_ids: Collection[str]
) -> list[str]:
    errors = []
    if answer.columns != ["query_id", "answer"]:
        errors.append(f"columns {answer.columns}")
        return errors
    if answer.schema["query_id"] != pl.String or answer.schema["answer"] != pl.String:
        errors.append(f"schema {dict(answer.schema)}")
        return errors
    ids = answer["query_id"].to_list()
    if len(ids) != len(set(ids)):
        errors.append("duplicate query_id")
    if set(ids) != set(query_ids):
        errors.append(
            f"query_id mismatch: {len(set(query_ids) - set(ids))} missing, "
            f"{len(set(ids) - set(query_ids))} extra"
        )
    if any(q is None or len(q) != ID_LENGTH for q in ids):
        errors.append("query_id length != 16")
    items = set(item_ids)
    for query_id, text in zip(ids, answer["answer"].to_list(), strict=True):
        found = (text or "").split(" ")
        if not text:
            errors.append(f"{query_id}: empty answer")
        elif len(found) > ANSWER_K:
            errors.append(f"{query_id}: {len(found)} items")
        elif len(set(found)) != len(found):
            errors.append(f"{query_id}: duplicate items")
        elif any(not ITEM_ID_RE.fullmatch(i) or i not in items for i in found):
            errors.append(f"{query_id}: item_id outside corpus or malformed")
    return errors


def read_answer(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, schema={"query_id": pl.String, "answer": pl.String})


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()
