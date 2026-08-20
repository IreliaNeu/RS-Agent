"""Paired Judge and Knowledge Bridge comparisons over verified exports."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rs_agent.evaluation.bootstrap import bootstrap_metric_intervals


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> Optional[float]:
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = set(left) | set(right)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def _candidate_key(row: Mapping[str, str], stage: str) -> Tuple[str, ...]:
    if stage == "rs_vqa":
        return (row["item_id"], row["question"], row["label"])
    return (row["item_id"], row["label"])


def compare_judge_exports(left_dir: Path, right_dir: Path, stage: str) -> dict:
    """Compare scores and selected labels produced by two Judge configurations."""
    if stage not in {"rs_cc", "rs_vqa"}:
        raise ValueError("stage must be rs_cc or rs_vqa")
    filename = "caption_scores.csv" if stage == "rs_cc" else "vqa_scores.csv"
    left_rows = {_candidate_key(row, stage): row for row in _read_csv(left_dir / filename)}
    right_rows = {
        _candidate_key(row, stage): row for row in _read_csv(right_dir / filename)
    }
    if set(left_rows) != set(right_rows):
        raise ValueError("Judge exports contain different candidate sets")
    left_scores = []
    right_scores = []
    score_differences = []
    for key in sorted(left_rows):
        left_score = left_rows[key].get("score", "")
        right_score = right_rows[key].get("score", "")
        if left_score != "" and right_score != "":
            left_scores.append(float(left_score))
            right_scores.append(float(right_score))
            score_differences.append(abs(float(left_score) - float(right_score)))

    group_size = 3 if stage == "rs_vqa" else 2
    groups = sorted({key[: group_size - 1] for key in left_rows})
    left_labels = []
    right_labels = []
    for group in groups:
        left_selected = [
            key[-1]
            for key, row in left_rows.items()
            if key[:-1] == group and row.get("selected") == "True"
        ]
        right_selected = [
            key[-1]
            for key, row in right_rows.items()
            if key[:-1] == group and row.get("selected") == "True"
        ]
        if len(left_selected) != 1 or len(right_selected) != 1:
            raise ValueError("each Judge export must select exactly one candidate per group")
        left_labels.append(left_selected[0])
        right_labels.append(right_selected[0])
    return {
        "stage": stage,
        "candidate_count": len(left_rows),
        "scored_pair_count": len(left_scores),
        "selection_group_count": len(groups),
        "selection_agreement": (
            sum(a == b for a, b in zip(left_labels, right_labels)) / len(groups)
            if groups
            else None
        ),
        "selection_cohen_kappa": _cohen_kappa(left_labels, right_labels),
        "score_pearson": _pearson(left_scores, right_scores),
        "score_mean_absolute_difference": (
            statistics.fmean(score_differences) if score_differences else None
        ),
    }


def _selected_vqa(export_dir: Path) -> Dict[Tuple[str, str], dict]:
    rows = [
        row
        for row in _read_csv(export_dir / "vqa_scores.csv")
        if row.get("selected") == "True"
    ]
    output = {(row["item_id"], row["question"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("selected VQA rows are not unique by item and question")
    return output


def compare_knowledge_bridge_exports(
    without_dir: Path,
    with_dir: Path,
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compare aligned VQA runs; delta is Knowledge Bridge minus no bridge."""
    without = _selected_vqa(without_dir)
    with_bridge = _selected_vqa(with_dir)
    if set(without) != set(with_bridge):
        raise ValueError("Knowledge Bridge exports contain different questions")
    deltas = []
    answer_matches = []
    model_matches = []
    for key in sorted(without):
        left = without[key]
        right = with_bridge[key]
        if left.get("score") == "" or right.get("score") == "":
            continue
        deltas.append(float(right["score"]) - float(left["score"]))
        answer_matches.append(left.get("text", "") == right.get("text", ""))
        model_matches.append(left.get("model_id", "") == right.get("model_id", ""))
    intervals = bootstrap_metric_intervals(
        {"judge_score_delta": deltas},
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    interval = intervals.get("judge_score_delta")
    return {
        "paired_question_count": len(deltas),
        "delta_definition": "with Knowledge Bridge minus without Knowledge Bridge",
        "judge_score_delta": interval.model_dump(mode="json") if interval else None,
        "exact_answer_agreement": (
            sum(answer_matches) / len(answer_matches) if answer_matches else None
        ),
        "selected_model_agreement": (
            sum(model_matches) / len(model_matches) if model_matches else None
        ),
        "bootstrap": {
            "samples": bootstrap_samples,
            "confidence": confidence,
            "seed": seed,
        },
    }


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("output already exists: {}".format(path))
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
