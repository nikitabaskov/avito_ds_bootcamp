"""Пакетное построение эмбеддингов из заранее загруженных локальных весов."""

import argparse
import time

import torch

from candgen.core.data import SEED, load_corpus
from candgen.core.dense import (
    DenseConfig,
    config_for,
    embed_corpus,
    embedding_dir,
    load_model,
    passage_texts,
    truncation_share,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["split", "benchmark"], default="split")
    parser.add_argument("--model", default=DenseConfig.model)
    parser.add_argument("--params-chars", type=int, default=DenseConfig.params_chars)
    parser.add_argument("--max-seq-length", type=int, default=DenseConfig.max_seq_length)
    parser.add_argument("--batch-size", type=int, default=DenseConfig.batch_size)
    parser.add_argument("--truncation-sample", type=int, default=5_000)
    args = parser.parse_args()
    config = config_for(
        args.model,
        params_chars=args.params_chars,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )

    corpus = load_corpus(args.corpus)
    texts = passage_texts(corpus, config.params_chars)
    out_dir = embedding_dir(args.corpus, config)

    t = time.perf_counter()
    model = load_model(config)
    load_s = time.perf_counter() - t
    share = truncation_share(model, texts, args.truncation_sample, SEED)
    print(f"{len(texts)} passages -> {out_dir}; truncated share {share:.3f}")

    embed_corpus(
        model,
        texts,
        corpus["item_id"].to_list(),
        out_dir,
        config,
        {
            "corpus": args.corpus,
            "device": str(model.device),
            "dtype": "float16" if model.device.type == "cuda" else "float32",
            "truncated_share": share,
            "truncation_sample": args.truncation_sample,
            "model_load_s": load_s,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else None,
        },
    )
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
