import argparse
import dataclasses
import time

from candgen.core.bm25 import BM25Config, BM25Retriever, item_documents, query_texts
from candgen.core.evaluation import POOL_KS, recall_report
from candgen.scripts.common import load_eval, peak_rss_gb, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["dev", "test"], default="dev")
    parser.add_argument("--title-repeat", type=int, default=1)
    parser.add_argument("--query-filters", action="store_true")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    config = BM25Config(
        k1=args.k1, b=args.b, title_repeat=args.title_repeat, query_filters=args.query_filters
    )

    corpus, queries, seen_items = load_eval(args.part)
    item_ids = corpus["item_id"].to_list()

    timings = {}
    t = time.perf_counter()
    retriever = BM25Retriever(config)
    retriever.index(item_documents(corpus, config.title_repeat))
    timings["index_s"] = time.perf_counter() - t

    t = time.perf_counter()
    rows, _ = retriever.search(query_texts(queries, config.query_filters), k=max(POOL_KS))
    timings["search_s"] = time.perf_counter() - t
    candidates = [[item_ids[r] for r in row if r >= 0] for row in rows]

    report = {
        "channel": "bm25",
        "part": args.part,
        "config": dataclasses.asdict(config),
        "corpus_items": corpus.height,
        **recall_report(queries, candidates, seen_items),
        "timings": timings,
        "peak_rss_gb": peak_rss_gb(),
    }
    write_report(f"bm25_{args.part}_{config.tag()}", report)


if __name__ == "__main__":
    main()
