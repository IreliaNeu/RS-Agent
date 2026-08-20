"""Candidate-level caption metrics and bootstrap summaries."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from rs_agent.evaluation.bootstrap import bootstrap_metric_intervals
from rs_agent.evaluation.reference_metrics import CaptionReference, evaluate_caption

METRIC_NAMES = ("bleu_1", "bleu_4", "rouge_l", "change_flag_accuracy")


def caption_metric_values(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    return {
        "bleu_1": [float(record["bleu_1"]) for record in records],
        "bleu_4": [float(record["bleu_4"]) for record in records],
        "rouge_l": [float(record["rouge_l"]) for record in records],
        "change_flag_accuracy": [
            float(bool(record["change_flag_match"])) for record in records
        ],
    }


def build_candidate_metric_rows(
    caption_rows: Sequence[Mapping[str, Any]],
    references: Mapping[str, CaptionReference],
) -> List[Dict[str, Any]]:
    output = []
    for candidate in caption_rows:
        reference = references.get(str(candidate["item_id"]))
        matched = reference is not None
        metric = None
        if matched and bool(candidate["success"]):
            metric = evaluate_caption(
                str(candidate["item_id"]), str(candidate.get("text") or ""), reference
            )
        output.append(
            {
                "item_id": candidate["item_id"],
                "dataset": candidate.get("dataset", "unspecified"),
                "change_type": candidate.get("change_type", "unspecified"),
                "label": candidate["label"],
                "input_mode": candidate.get("input_mode", "text_only"),
                "model_name": candidate["model_name"],
                "provider": candidate["provider"],
                "model_id": candidate["model_id"],
                "success": candidate["success"],
                "selected": candidate["selected"],
                "judge_score": candidate.get("score"),
                "reference_matched": matched,
                "reference_count": len(reference.references) if reference else None,
                "change_flag": reference.change_flag if reference else None,
                "bleu_1": metric.bleu_1 if metric else None,
                "bleu_4": metric.bleu_4 if metric else None,
                "rouge_l": metric.rouge_l if metric else None,
                "change_flag_match": metric.change_flag_match if metric else None,
                "text": candidate.get("text", ""),
            }
        )
    return output


def _group_key(row: Mapping[str, Any], scope: str) -> Tuple[str, ...]:
    if scope == "input_mode":
        return (str(row["input_mode"]),)
    return (
        str(row["model_name"]),
        str(row["provider"]),
        str(row["model_id"]),
        str(row["input_mode"]),
    )


def summarize_candidate_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> List[Dict[str, Any]]:
    output = []
    for scope in ("input_mode", "model"):
        groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[_group_key(row, scope)].append(row)
        for key, group in sorted(groups.items()):
            measured = [row for row in group if row.get("bleu_1") is not None]
            score_values = [
                float(row["judge_score"])
                for row in group
                if row.get("judge_score") is not None
            ]
            intervals = bootstrap_metric_intervals(
                caption_metric_values(measured),
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
            )
            if scope == "input_mode":
                model_name = provider = model_id = ""
                input_mode = key[0]
            else:
                model_name, provider, model_id, input_mode = key
            record: Dict[str, Any] = {
                "scope": scope,
                "input_mode": input_mode,
                "model_name": model_name,
                "provider": provider,
                "model_id": model_id,
                "candidate_count": len(group),
                "metric_count": len(measured),
                "selected_count": sum(bool(row["selected"]) for row in group),
                "mean_judge_score": (
                    statistics.fmean(score_values) if score_values else None
                ),
            }
            for metric_name in METRIC_NAMES:
                interval = intervals.get(metric_name)
                record["{}_mean".format(metric_name)] = (
                    interval.mean if interval else None
                )
                record["{}_lower".format(metric_name)] = (
                    interval.lower if interval else None
                )
                record["{}_upper".format(metric_name)] = (
                    interval.upper if interval else None
                )
            output.append(record)
    return output
