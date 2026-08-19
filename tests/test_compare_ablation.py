import csv
import json
from pathlib import Path

import pytest

from rs_agent.evaluation.ablation import compare_exports


def write_export(root: Path, fingerprint: str, bleu: float, score: float) -> None:
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "experiment_fingerprint": fingerprint,
                "caption_references_sha256": "r" * 64,
            }
        ),
        encoding="utf-8",
    )
    with (root / "caption_reference_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "change_flag",
                "selected_caption",
                "bleu_1",
                "bleu_4",
                "rouge_l",
                "change_flag_match",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "item-1",
                "change_flag": 1,
                "selected_caption": "caption",
                "bleu_1": bleu,
                "bleu_4": 0.0,
                "rouge_l": bleu,
                "change_flag_match": True,
            }
        )
    with (root / "caption_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "selected",
                "score",
                "latency_ms",
                "model_name",
                "input_mode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "item-1",
                "selected": True,
                "score": score,
                "latency_ms": 10,
                "model_name": "model",
                "input_mode": "text_only",
            }
        )


def test_compare_exports_uses_paired_enhancement_minus_baseline_delta(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    enhancement = tmp_path / "enhancement"
    write_export(baseline, "a" * 64, 0.25, 7)
    write_export(enhancement, "b" * 64, 0.75, 9)

    summary = compare_exports(
        baseline,
        enhancement,
        tmp_path / "comparison",
        baseline_name="text",
        enhancement_name="mixed",
        bootstrap_samples=50,
        confidence=0.95,
        seed=5,
    )

    assert summary["metric_deltas"]["bleu_1"]["mean"] == pytest.approx(0.5)
    assert summary["metric_deltas"]["judge_score"]["mean"] == pytest.approx(2.0)
    assert summary["paired_item_count"] == 1
