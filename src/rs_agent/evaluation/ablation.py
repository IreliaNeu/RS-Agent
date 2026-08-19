"""Paired comparison utilities for RS-CC ablation exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from rs_agent.evaluation.bootstrap import bootstrap_metric_intervals

METRIC_COLUMNS = {
    "bleu_1": "bleu_1",
    "bleu_4": "bleu_4",
    "rouge_l": "rouge_l",
    "change_flag_accuracy": "change_flag_match",
    "judge_score": "judge_score",
    "selected_latency_ms": "selected_latency_ms",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> float:
    if value in {"True", "False"}:
        return float(value == "True")
    return float(value)


def _selected_records(export_dir: Path) -> Dict[str, dict]:
    references = {
        row["item_id"]: row
        for row in _read_csv(export_dir / "caption_reference_metrics.csv")
    }
    selected = {
        row["item_id"]: row
        for row in _read_csv(export_dir / "caption_scores.csv")
        if row["selected"] == "True"
    }
    if set(references) != set(selected):
        raise ValueError("selected captions and reference metrics have different item sets")
    output = {}
    for item_id, record in references.items():
        candidate = selected[item_id]
        output[item_id] = {
            **record,
            "judge_score": candidate["score"],
            "selected_latency_ms": candidate["latency_ms"],
            "selected_model": candidate["model_name"],
            "selected_input_mode": candidate["input_mode"],
        }
    return output


def compare_exports(
    baseline_dir: Path,
    enhancement_dir: Path,
    output_dir: Path,
    *,
    baseline_name: str,
    enhancement_name: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict:
    """Compare aligned exports and persist paired metric deltas."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = _read_json(baseline_dir / "summary.json")
    enhancement_summary = _read_json(enhancement_dir / "summary.json")
    if baseline_summary.get("caption_references_sha256") != enhancement_summary.get(
        "caption_references_sha256"
    ):
        raise ValueError("exports use different reference manifests")
    baseline = _selected_records(baseline_dir)
    enhancement = _selected_records(enhancement_dir)
    if set(baseline) != set(enhancement):
        raise ValueError("paired exports must contain identical item IDs")

    paired_rows = []
    deltas: Dict[str, List[float]] = {name: [] for name in METRIC_COLUMNS}
    for item_id in sorted(baseline):
        left = baseline[item_id]
        right = enhancement[item_id]
        row = {
            "item_id": item_id,
            "change_flag": left["change_flag"],
            "baseline_caption": left["selected_caption"],
            "enhancement_caption": right["selected_caption"],
            "baseline_model": left["selected_model"],
            "enhancement_model": right["selected_model"],
            "baseline_input_mode": left["selected_input_mode"],
            "enhancement_input_mode": right["selected_input_mode"],
        }
        for metric_name, column in METRIC_COLUMNS.items():
            baseline_value = _as_float(left[column])
            enhancement_value = _as_float(right[column])
            delta = enhancement_value - baseline_value
            row["baseline_{}".format(metric_name)] = baseline_value
            row["enhancement_{}".format(metric_name)] = enhancement_value
            row["delta_{}".format(metric_name)] = delta
            deltas[metric_name].append(delta)
        paired_rows.append(row)

    intervals = bootstrap_metric_intervals(
        deltas,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    summary = {
        "baseline_name": baseline_name,
        "enhancement_name": enhancement_name,
        "baseline_fingerprint": baseline_summary["experiment_fingerprint"],
        "enhancement_fingerprint": enhancement_summary["experiment_fingerprint"],
        "caption_references_sha256": baseline_summary["caption_references_sha256"],
        "paired_item_count": len(paired_rows),
        "delta_definition": "enhancement minus baseline",
        "bootstrap": {
            "method": "paired percentile bootstrap of the arithmetic mean delta",
            "samples": bootstrap_samples,
            "confidence": confidence,
            "seed": seed,
        },
        "metric_deltas": {
            name: interval.model_dump(mode="json")
            for name, interval in intervals.items()
        },
    }
    with (output_dir / "summary.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    columns = [
        "item_id",
        "change_flag",
        "baseline_caption",
        "enhancement_caption",
        "baseline_model",
        "enhancement_model",
        "baseline_input_mode",
        "enhancement_input_mode",
    ]
    for metric_name in METRIC_COLUMNS:
        columns.extend(
            [
                "baseline_{}".format(metric_name),
                "enhancement_{}".format(metric_name),
                "delta_{}".format(metric_name),
            ]
        )
    with (output_dir / "paired_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paired_rows)
    return summary
