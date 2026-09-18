import numpy as np
import polars as pl
import torch

from candgen.core.history import HistoryView

MICROCAT_MODES = ("none", "neighbors")
MICROCAT_FEATURES = {
    "none": [],
    "neighbors": ["mc_prob", "mc_ratio", "mc_entropy", "mc_top_sim"],
}
NEIGHBORS = 40
TEMPERATURE = 0.05
BLOCK_SIZE = 1024


def text_microcats(pairs: pl.DataFrame) -> pl.DataFrame:
    counts = pairs.drop_nulls("item_microcat_id").group_by("query_text", "item_microcat_id").len()
    return counts.select(
        "query_text",
        "item_microcat_id",
        share=pl.col("len") / pl.col("len").sum().over("query_text"),
    )


class MicrocatIndex:
    def __init__(self, texts: list[str], vectors: np.ndarray, device: str):
        self.texts = pl.Series("query_text", texts)
        self.matrix = torch.from_numpy(vectors).to(device)

    def neighbors(self, vectors: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.matrix[torch.from_numpy(allowed).to(self.matrix.device)]
        k = min(NEIGHBORS, len(allowed))
        rows, sims = [], []
        for start in range(0, len(vectors), BLOCK_SIZE):
            block = torch.from_numpy(vectors[start : start + BLOCK_SIZE]).to(matrix.device)
            top_sims, top_rows = torch.topk(block @ matrix.T, k=k, dim=1)
            rows.append(top_rows.cpu().numpy())
            sims.append(top_sims.float().cpu().numpy())
        return allowed[np.concatenate(rows)], np.concatenate(sims)

    def distribution(self, view: HistoryView, vectors: np.ndarray) -> pl.DataFrame:
        queries, pairs = view
        shares = text_microcats(pairs)
        allowed = np.flatnonzero(self.texts.is_in(shares["query_text"].unique().implode()))
        q = queries["q"].to_numpy()
        if q.size == 0 or allowed.size == 0:
            return pl.DataFrame(
                schema={
                    "q": pl.Int32,
                    "item_microcat_id": shares.schema["item_microcat_id"],
                    "mc_prob": pl.Float64,
                    "mc_max": pl.Float64,
                    "mc_entropy": pl.Float64,
                    "mc_top_sim": pl.Float64,
                }
            )
        rows, sims = self.neighbors(vectors[q], allowed)
        weights = np.exp((sims - sims[:, :1]) / TEMPERATURE)
        hits = pl.DataFrame(
            {
                "q": np.repeat(q, rows.shape[1]).astype(np.int32),
                "query_text": self.texts.gather(rows.ravel()),
                "w": (weights / weights.sum(axis=1, keepdims=True)).ravel(),
            }
        )
        probs = (
            hits.join(shares, on="query_text")
            .group_by("q", "item_microcat_id")
            .agg(mc_prob=(pl.col("w") * pl.col("share")).sum())
        )
        stats = probs.group_by("q").agg(
            mc_max=pl.col("mc_prob").max(),
            mc_entropy=-(pl.col("mc_prob") * pl.col("mc_prob").log()).sum(),
        )
        top = pl.DataFrame({"q": q.astype(np.int32), "mc_top_sim": sims[:, 0].astype(np.float64)})
        return probs.join(stats, on="q").join(top, on="q")


def add_microcat_features(
    frame: pl.DataFrame,
    views: list[HistoryView],
    index: MicrocatIndex,
    vectors: np.ndarray,
    item_microcats: pl.Series,
) -> pl.DataFrame:
    dist = pl.concat([index.distribution(view, vectors) for view in views])
    per_query = dist.select("q", "mc_max", "mc_entropy", "mc_top_sim").unique("q")
    return (
        frame.with_columns(item_microcat_id=item_microcats.gather(frame["row"]))
        .join(
            dist.select("q", "item_microcat_id", "mc_prob"),
            on=["q", "item_microcat_id"],
            how="left",
        )
        .join(per_query, on="q", how="left")
        .with_columns(
            mc_prob=pl.col("mc_prob").fill_null(0.0).cast(pl.Float32),
            mc_ratio=(pl.col("mc_prob").fill_null(0.0) / pl.col("mc_max")).cast(pl.Float32),
            mc_entropy=pl.col("mc_entropy").cast(pl.Float32),
            mc_top_sim=pl.col("mc_top_sim").cast(pl.Float32),
        )
        .select(*frame.columns, *MICROCAT_FEATURES["neighbors"])
        .sort("q", "rrf_rank")
    )
