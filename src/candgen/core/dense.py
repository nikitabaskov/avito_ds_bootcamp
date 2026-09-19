import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from candgen.core.data import ARTIFACTS_DIR
from candgen.core.retrieval import Groups, Hits, order_hits, search_groups, search_local

EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
CHUNK_SIZE = 50_000
REVISIONS = {
    "intfloat/multilingual-e5-base": "d128750597153bb5987e10b1c3493a34e5a4502a",
    "deepvk/USER-base": "e8446472f6024df155a04b2f0911b6044cabc51f",
}


@dataclass(frozen=True)
class DenseConfig:
    model: str = "intfloat/multilingual-e5-base"
    revision: str | None = REVISIONS["intfloat/multilingual-e5-base"]
    max_seq_length: int = 512
    params_chars: int = 500
    batch_size: int = 64
    query_filters: bool = True

    def passage_tag(self) -> str:
        return f"{self.model.split('/')[-1]}_len{self.max_seq_length}_p{self.params_chars}"


def config_for(model: str, **kwargs) -> DenseConfig:
    return DenseConfig(model=model, revision=REVISIONS.get(model), **kwargs)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clean(column: str) -> pl.Expr:
    text = pl.col(column).fill_null("").str.replace_all(r"\s+", " ").str.strip_chars()
    return pl.when(text != "").then(text)


def passage_texts(items: pl.DataFrame, params_chars: int) -> list[str]:
    body = pl.concat_str(
        [
            _clean("item_title_raw"),
            _clean("item_infm_params_text").str.slice(0, params_chars),
            _clean("item_description_raw"),
        ],
        separator="\n",
        ignore_nulls=True,
    )
    return items.select(pl.lit("passage: ") + body).to_series().to_list()


def query_texts(queries: pl.DataFrame, with_filters: bool) -> list[str]:
    parts = [_clean("query_text")]
    if with_filters:
        parts.append(_clean("search_infm_params_text"))
    body = pl.concat_str(parts, separator=". ", ignore_nulls=True).fill_null("")
    return queries.select(pl.lit("query: ") + body).to_series().to_list()


def load_model(config: DenseConfig, device: str | None = None) -> SentenceTransformer:
    device = device or default_device()
    model = SentenceTransformer(config.model, revision=config.revision, device=device)
    if model.max_seq_length != config.max_seq_length:
        model.max_seq_length = config.max_seq_length
    if device == "cuda":
        model.half()
    return model


def encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def truncation_share(model: SentenceTransformer, texts: list[str], sample: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(texts), size=min(sample, len(texts)), replace=False)
    lengths = [len(ids) for ids in model.tokenizer([texts[i] for i in picked])["input_ids"]]
    return float(np.mean(np.array(lengths) > model.max_seq_length))


def ids_digest(item_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(item_ids).encode()).hexdigest()


def embedding_dir(corpus: str, config: DenseConfig) -> Path:
    return EMBEDDINGS_DIR / f"{corpus}_{config.passage_tag()}"


def embed_corpus(
    model: SentenceTransformer,
    texts: list[str],
    item_ids: list[str],
    out_dir: Path,
    config: DenseConfig,
    extra_meta: dict,
) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    chunks = []
    for i, start in enumerate(range(0, len(texts), CHUNK_SIZE)):
        path = out_dir / f"chunk_{i:04d}.npy"
        if not path.exists():
            part = encode(model, texts[start : start + CHUNK_SIZE], config.batch_size)
            tmp = out_dir / f"chunk_{i:04d}.tmp.npy"
            np.save(tmp, part.astype(np.float16))
            tmp.rename(path)
        chunks.append(np.load(path))
    matrix = np.concatenate(chunks)
    np.save(out_dir / "embeddings.npy", matrix)
    meta = {
        "config": asdict(config),
        "items": len(item_ids),
        "dim": int(matrix.shape[1]),
        "item_ids_sha256": ids_digest(item_ids),
        "encode_s": time.perf_counter() - started,
        **extra_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    for path in out_dir.glob("chunk_*.npy"):
        path.unlink()
    return matrix.astype(np.float32)


def load_embeddings(out_dir: Path, item_ids: list[str]) -> np.ndarray:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no embeddings in {out_dir}; run candgen.scripts.embed_corpus")
    meta = json.loads(meta_path.read_text())
    if meta["item_ids_sha256"] != ids_digest(item_ids):
        raise ValueError(f"embeddings in {out_dir} do not match corpus item_id order")
    return np.load(out_dir / "embeddings.npy").astype(np.float32)


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, device: str | None = None, block_size: int = 256):
        self.device = device or default_device()
        self.matrix = torch.from_numpy(embeddings).to(self.device)
        self.block_size = block_size

    def search(self, queries: np.ndarray, k: int, items: np.ndarray | None = None) -> Hits:
        matrix = self.matrix
        if items is not None:
            matrix = matrix[torch.from_numpy(items).to(self.device)]
        k = min(k, matrix.shape[0])
        rows, scores = [], []
        for start in range(0, len(queries), self.block_size):
            block = torch.from_numpy(queries[start : start + self.block_size]).to(self.device)
            top_scores, top_rows = torch.topk(block @ matrix.T, k=k, dim=1)
            rows.append(top_rows.cpu().numpy().astype(np.int64))
            scores.append(top_scores.float().cpu().numpy())
        found = np.concatenate(rows)
        if items is not None:
            found = items[found]
        return order_hits(found, np.concatenate(scores))

    def search_local(
        self,
        queries: np.ndarray,
        query_locations: np.ndarray,
        item_locations: np.ndarray,
        k: int,
    ) -> Hits:
        return search_local(
            lambda q, items, depth: self.search(queries[q], depth, items),
            query_locations,
            item_locations,
            k,
        )

    def search_groups(self, queries: np.ndarray, groups: Groups, k: int) -> Hits:
        return search_groups(
            lambda q, items, depth: self.search(queries[q], depth, items), groups, len(queries), k
        )
