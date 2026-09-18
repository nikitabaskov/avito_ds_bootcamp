import json
from pathlib import Path

import polars as pl

from candgen.core.evaluation import COMPARISON_SLICES, compare_per_query
from candgen.scripts.common import EXPERIMENTS_DIR

REPORTS_DIR = Path("reports")
DECISIONS_PATH = REPORTS_DIR / "decisions.json"
B0 = "EXP-000/b0"
VERDICTS = {"better": "лучше", "worse": "хуже", "undetermined": "не определено"}
SLICE_TITLES = {
    "no_center": "Без центра",
    "has_center": "С центром",
    "filters_set": "С фильтрами",
    "filters_empty": "Без фильтров",
    "words_1": "Однословные",
    "other_location": "Позитив в другой локации",
}


def pp(value: float) -> str:
    return f"{value * 100:+.2f}".replace(".", ",")


def interval(result: dict | None) -> str:
    if not result:
        return "—"
    low, high = result["ci95"]
    mark = "" if result["verdict"] == "undetermined" else "**"
    return f"{mark}{pp(result['diff'])}{mark} [{pp(low)}; {pp(high)}]"


def runs() -> list[str]:
    return sorted(
        str(path.parent.relative_to(EXPERIMENTS_DIR))
        for path in EXPERIMENTS_DIR.glob("EXP-*/*/report.json")
    )


def comparison(name: str, reference: str) -> dict | None:
    if name == reference or not (EXPERIMENTS_DIR / reference / "per_query.parquet").exists():
        return None
    return compare_per_query(
        pl.read_parquet(EXPERIMENTS_DIR / name / "per_query.parquet"),
        pl.read_parquet(EXPERIMENTS_DIR / reference / "per_query.parquet"),
    )


def summarize(name: str, decisions: dict) -> dict:
    report = json.loads((EXPERIMENTS_DIR / name / "report.json").read_text())
    train = report.get("train") or {}
    return {
        "experiment": name,
        "parent": report["parent"],
        "recall@50": report["recall@50"],
        "pool_recall": report["pool_recall"],
        "pool_size_mean": report["pool_size"]["mean"],
        "trees": train.get("best_iteration", report["model"].get("best_iteration")),
        "git": report["git"],
        "decision": decisions.get(name, ""),
        "vs_parent": comparison(name, report["parent"]),
        "vs_b0": comparison(name, B0),
    }


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Сводка экспериментов",
        "",
        (
            "Собрано `uv run python -m candgen.scripts.summary` из `data/artifacts/experiments/`. "
            "Разности — в процентных пунктах macro Recall@50 на 2 452 dev-запросах; "
            "CI95 — парный bootstrap по запросам, 5 000 повторов, seed 42. "
            "Жирным выделены разности, чей интервал не содержит ноль. "
            "Вердикт — только статистика; решение учитывает ещё срезы, ресурсы и устойчивость обучения."
        ),
        "",
        "## Итог по вариантам",
        "",
        (
            "| Эксперимент | Родитель | R@50 | К родителю, п.п. [CI95] | Вердикт | К B0, п.п. [CI95]"
            " | Полнота пула | Полнота к родителю [CI95] | Деревьев | Решение |"
        ),
        "| --- | --- | ---: | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        parent = row["vs_parent"]
        lines.append(
            f"| {row['experiment']} | {row['parent']} | {row['recall@50']:.4f}".replace(".", ",")
            + f" | {interval(parent)} | {VERDICTS[parent['verdict']] if parent else '—'}"
            + f" | {interval(row['vs_b0'])} | {row['pool_recall']:.4f}".replace(".", ",")
            + f" | {interval(parent['pool_recall'] if parent else None)}"
            + f" | {row['trees'] if row['trees'] is not None else '—'} | {row['decision']} |"
        )
    lines += [
        "",
        "## Срезы: разность к родителю, п.п. [CI95]",
        "",
        "| Эксперимент | " + " | ".join(SLICE_TITLES[s] for s in COMPARISON_SLICES) + " |",
        "| --- |" + " --- |" * len(COMPARISON_SLICES),
    ]
    for row in rows:
        parent = row["vs_parent"]
        if not parent:
            continue
        cells = [interval(parent["slices"][s]) for s in COMPARISON_SLICES]
        lines.append(f"| {row['experiment']} | " + " | ".join(cells) + " |")
    sizes = next((r["vs_parent"]["slices"] for r in rows if r["vs_parent"]), None)
    if sizes:
        lines += [
            "",
            "Размер срезов, запросов: "
            + ", ".join(f"{SLICE_TITLES[s]} — {v['n']}" for s, v in sizes.items())
            + ".",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    decisions = json.loads(DECISIONS_PATH.read_text()) if DECISIONS_PATH.exists() else {}
    rows = [summarize(name, decisions) for name in runs()]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "experiments.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    (REPORTS_DIR / "experiments.md").write_text(markdown(rows))
    print(markdown(rows))


if __name__ == "__main__":
    main()
