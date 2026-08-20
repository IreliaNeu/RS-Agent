import csv
from pathlib import Path

from rs_agent.evaluation.comparisons import (
    compare_judge_exports,
    compare_knowledge_bridge_exports,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def caption_rows(selected: str, offset: int = 0) -> list[dict]:
    return [
        {
            "item_id": "pair-1",
            "label": label,
            "score": score + offset,
            "selected": str(label == selected),
        }
        for label, score in (("A", 8), ("B", 7))
    ]


def test_compare_judges_reports_score_and_selection_agreement(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_rows(left / "caption_scores.csv", caption_rows("A"))
    write_rows(right / "caption_scores.csv", caption_rows("A", 1))

    summary = compare_judge_exports(left, right, "rs_cc")

    assert summary["selection_agreement"] == 1.0
    assert summary["score_mean_absolute_difference"] == 1.0


def test_compare_knowledge_bridge_pairs_by_question_text(tmp_path: Path) -> None:
    without = tmp_path / "without"
    with_bridge = tmp_path / "with"
    base = {
        "item_id": "pair-1",
        "question": "What changed?",
        "label": "A",
        "selected": "True",
        "model_id": "model-a",
        "text": "A building appeared.",
    }
    write_rows(without / "vqa_scores.csv", [{**base, "score": 7}])
    write_rows(with_bridge / "vqa_scores.csv", [{**base, "score": 9}])

    summary = compare_knowledge_bridge_exports(
        without, with_bridge, bootstrap_samples=20, seed=3
    )

    assert summary["paired_question_count"] == 1
    assert summary["judge_score_delta"]["mean"] == 2.0
