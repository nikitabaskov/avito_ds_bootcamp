import json

import polars as pl

from candgen.core.data import (
    EVAL_QUERIES_PER_PART,
    SEED,
    SPLIT_DIR,
    assign_parts,
    build_contexts,
    build_corpus,
    load_train,
    sample_eval_queries,
)


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train()
    parts = assign_parts(train["query_text"], SEED)
    parts.write_parquet(SPLIT_DIR / "text_parts.parquet")

    pairs = train.join(parts, on="query_text", how="left", validate="m:1")
    meta: dict = {"seed": SEED, "train_rows": train.height, "parts": {}}
    for part in ("train", "dev", "test"):
        contexts = build_contexts(pairs.filter(pl.col("part") == part))
        contexts.write_parquet(SPLIT_DIR / f"contexts_{part}.parquet")
        meta["parts"][part] = {
            "texts": contexts["query_text"].n_unique(),
            "contexts": contexts.height,
            "positive_pairs": int(contexts["item_ids"].list.len().sum()),
        }
        if part != "train":
            sample = sample_eval_queries(contexts, EVAL_QUERIES_PER_PART, SEED)
            sample.write_parquet(SPLIT_DIR / f"eval_{part}.parquet")
            meta["parts"][part]["eval_queries"] = sample.height

    texts = {
        p: set(parts.filter(pl.col("part") == p)["query_text"]) for p in ("train", "dev", "test")
    }
    assert not (
        texts["train"] & texts["dev"]
        or texts["train"] & texts["test"]
        or texts["dev"] & texts["test"]
    )

    corpus = build_corpus(train)
    corpus.write_parquet(SPLIT_DIR / "corpus.parquet")
    meta["corpus_items"] = corpus.height

    (SPLIT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
