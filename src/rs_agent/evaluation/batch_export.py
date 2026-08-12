"""Checksum-verified exports from resumable batch state and immutable artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.evaluation.reference_metrics import (
    REFERENCE_METRICS_VERSION,
    CaptionMetricRecord,
    evaluate_caption,
    load_reference_manifest,
    mean_caption_metrics,
)
from rs_agent.experiments.identity import sha256_file, source_tree_sha256
from rs_agent.experiments.state import BatchRunState, ItemStatus


def read_artifact(path: str, expected_type: str) -> ArtifactEnvelope:
    artifact = JsonArtifactStore(Path(".")).read(Path(path))
    if artifact.artifact_type != expected_type:
        raise ValueError(
            "expected {} artifact, got {} at {}".format(
                expected_type, artifact.artifact_type, path
            )
        )
    return artifact


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _request_fields(
    response: Optional[Dict[str, Any]],
    fallback_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = response or {}
    usage = response.get("usage") or {}
    telemetry = response.get("telemetry") or fallback_telemetry or {}
    status_codes = telemetry.get("attempt_status_codes") or []
    outcomes = telemetry.get("attempt_outcomes") or []
    return {
        "attempts": int(telemetry.get("attempts", 1)),
        "latency_ms": float(telemetry.get("latency_ms", 0.0)),
        "status_code": telemetry.get("status_code"),
        "attempt_status_codes": json.dumps(status_codes, separators=(",", ":")),
        "attempt_outcomes": json.dumps(outcomes, separators=(",", ":")),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _request_row(
    *,
    item_id: str,
    stage: str,
    role: str,
    model_name: str,
    provider: str,
    model_id: str,
    success: bool,
    error: Optional[str],
    response: Optional[Dict[str, Any]],
    telemetry: Optional[Dict[str, Any]] = None,
    question_id: Optional[str] = None,
    label: Optional[str] = None,
    input_mode: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "stage": stage,
        "role": role,
        "question_id": question_id,
        "label": label,
        "input_mode": input_mode,
        "model_name": model_name,
        "provider": provider,
        "model_id": model_id,
        "success": success,
        "error": error,
        **_request_fields(response, telemetry),
    }


def _caption_data(
    item_id: str, result_payload: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    result_path = result_payload["source_artifacts"].get("rs_cc_result")
    if not result_path:
        return [], []
    caption_result = read_artifact(result_path, "rs_cc_result")
    source_artifacts = caption_result.payload["source_artifacts"]
    generation = read_artifact(
        source_artifacts["generation"], "caption_candidates"
    ).payload
    evaluation = read_artifact(
        source_artifacts["evaluation"], "caption_evaluation"
    ).payload
    scores = evaluation.get("scores", {})
    selected = evaluation["selection"]["selected_label"]
    judge_choice = evaluation.get("judge_choice") or None
    rows = []
    requests = []
    for candidate in generation["candidates"]:
        response = candidate.get("response")
        request_fields = _request_fields(response, candidate.get("telemetry"))
        rows.append(
            {
                "item_id": item_id,
                "label": candidate["label"],
                "input_mode": candidate.get("input_mode", "text_only"),
                "model_name": candidate["model"]["name"],
                "provider": candidate["model"]["provider"],
                "model_id": candidate["model"]["model"],
                "success": not bool(candidate.get("error")),
                "error": candidate.get("error"),
                "score": scores.get(candidate["label"]),
                "selected": candidate["label"] == selected,
                "judge_choice": candidate["label"] == judge_choice,
                **request_fields,
                "text": candidate.get("text", ""),
            }
        )
        requests.append(
            _request_row(
                item_id=item_id,
                stage="rs_cc",
                role="generator",
                label=candidate["label"],
                input_mode=candidate.get("input_mode", "text_only"),
                model_name=candidate["model"]["name"],
                provider=candidate["model"]["provider"],
                model_id=candidate["model"]["model"],
                success=not bool(candidate.get("error")),
                error=candidate.get("error"),
                response=response,
                telemetry=candidate.get("telemetry"),
            )
        )
    for role in ("selector", "evaluator"):
        response = evaluation.get("{}_response".format(role)) or {}
        requests.append(
            _request_row(
                item_id=item_id,
                stage="rs_cc",
                role=role,
                model_name=role,
                provider=str(response.get("provider", "")),
                model_id=str(response.get("model", "")),
                success=True,
                error=None,
                response=response,
            )
        )
    return rows, requests


def _vqa_data(
    item_id: str, result_payload: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    result_path = result_payload["source_artifacts"].get("rs_vqa_result")
    if not result_path:
        return [], []
    vqa_result = read_artifact(result_path, "rs_vqa_result")
    source_artifacts = vqa_result.payload["source_artifacts"]
    generation = read_artifact(
        source_artifacts["generation"], "vqa_candidates"
    ).payload
    evaluation = read_artifact(
        source_artifacts["evaluation"], "vqa_evaluation"
    ).payload
    evaluation_by_question = {
        record["question"]["question_id"]: record for record in evaluation["records"]
    }
    questions = {
        question["question_id"]: question for question in generation["questions"]
    }
    rows = []
    requests = []
    for candidate in generation["candidates"]:
        question_id = candidate["question_id"]
        question = questions[question_id]
        record = evaluation_by_question.get(question_id)
        scores = record.get("scores", {}) if record else {}
        selected = record["selection"]["selected_label"] if record else None
        judge_choice = (record.get("judge_choice") or None) if record else None
        response = candidate.get("response")
        request_fields = _request_fields(response, candidate.get("telemetry"))
        rows.append(
            {
                "item_id": item_id,
                "question_id": question_id,
                "question": question["text"],
                "question_source": question["source"],
                "label": candidate["label"],
                "model_name": candidate["model"]["name"],
                "provider": candidate["model"]["provider"],
                "model_id": candidate["model"]["model"],
                "success": not bool(candidate.get("error")),
                "error": candidate.get("error"),
                "score": scores.get(candidate["label"]),
                "selected": candidate["label"] == selected,
                "judge_choice": candidate["label"] == judge_choice,
                **request_fields,
                "text": candidate.get("text", ""),
            }
        )
        requests.append(
            _request_row(
                item_id=item_id,
                stage="rs_vqa",
                role="generator",
                question_id=question_id,
                label=candidate["label"],
                model_name=candidate["model"]["name"],
                provider=candidate["model"]["provider"],
                model_id=candidate["model"]["model"],
                success=not bool(candidate.get("error")),
                error=candidate.get("error"),
                response=response,
                telemetry=candidate.get("telemetry"),
            )
        )
    for record in evaluation["records"]:
        question_id = record["question"]["question_id"]
        for role in ("selector", "evaluator"):
            response = record.get("{}_response".format(role)) or {}
            requests.append(
                _request_row(
                    item_id=item_id,
                    stage="rs_vqa",
                    role=role,
                    question_id=question_id,
                    model_name=role,
                    provider=str(response.get("provider", "")),
                    model_id=str(response.get("model", "")),
                    success=True,
                    error=None,
                    response=response,
                )
            )
    return rows, requests


def _percentile_95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _model_summary(
    request_rows: List[Dict[str, Any]],
    caption_rows: List[Dict[str, Any]],
    vqa_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    score_lookup: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for stage, rows in (("rs_cc", caption_rows), ("rs_vqa", vqa_rows)):
        for row in rows:
            score_lookup[
                (stage, row["model_name"], row["provider"], row["model_id"])
            ].append(row)
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        grouped[
            (
                row["stage"],
                row["role"],
                row["model_name"],
                row["provider"],
                row["model_id"],
                row.get("input_mode") or "",
            )
        ].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        stage, role, name, provider, model_id, input_mode = key
        successful = sum(bool(row["success"]) for row in rows)
        latencies = [float(row["latency_ms"]) for row in rows]
        known_tokens = [
            int(row["total_tokens"])
            for row in rows
            if row.get("total_tokens") is not None
        ]
        attempts = sum(int(row["attempts"]) for row in rows)
        candidate_rows = score_lookup.get((stage, name, provider, model_id), [])
        scores = [
            float(row["score"])
            for row in candidate_rows
            if row.get("score") is not None
        ]
        output.append(
            {
                "stage": stage,
                "role": role,
                "model_name": name,
                "provider": provider,
                "model_id": model_id,
                "input_mode": input_mode,
                "request_count": len(rows),
                "successful_count": successful,
                "failure_count": len(rows) - successful,
                "success_rate": round(successful / len(rows), 6),
                "score_count": len(scores),
                "mean_score": round(statistics.fmean(scores), 6) if scores else None,
                "population_std": (
                    round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0
                ) if scores else None,
                "selected_count": sum(bool(row["selected"]) for row in candidate_rows),
                "judge_choice_count": sum(
                    bool(row["judge_choice"]) for row in candidate_rows
                ),
                "attempt_count": attempts,
                "retry_count": attempts - len(rows),
                "mean_latency_ms": round(statistics.fmean(latencies), 3),
                "p95_latency_ms": round(_percentile_95(latencies) or 0.0, 3),
                "known_usage_count": len(known_tokens),
                "total_tokens": sum(known_tokens) if known_tokens else None,
            }
        )
    return output


def _caption_metric_rows(
    item_rows: List[Dict[str, Any]], references_path: Optional[Path]
) -> Tuple[List[Dict[str, Any]], List[CaptionMetricRecord], int, Optional[str]]:
    if references_path is None:
        return [], [], 0, None
    references = load_reference_manifest(references_path)
    rows = []
    records = []
    missing = 0
    for item in item_rows:
        reference = references.get(item["item_id"])
        if reference is None:
            missing += 1
            continue
        metric = evaluate_caption(
            item["item_id"], item.get("selected_caption") or "", reference
        )
        records.append(metric)
        rows.append(
            {
                "item_id": item["item_id"],
                "split": reference.split,
                "reference_count": len(reference.references),
                "change_flag": reference.change_flag,
                "selected_caption": item.get("selected_caption") or "",
                **metric.model_dump(mode="json"),
            }
        )
    return rows, records, missing, sha256_file(references_path.resolve())


def export_batch(
    state: BatchRunState,
    output_dir: Path,
    references_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    item_rows = []
    failure_rows = []
    caption_rows: List[Dict[str, Any]] = []
    vqa_rows: List[Dict[str, Any]] = []
    request_rows: List[Dict[str, Any]] = []
    consensus_counts: Counter = Counter()
    mask_caption_counts: Counter = Counter()
    conflict_items = 0

    for item_id, item_state in state.items.items():
        if item_state.status != ItemStatus.COMPLETED:
            failure_rows.append(
                {
                    "item_id": item_id,
                    "status": item_state.status.value,
                    "attempts": item_state.attempts,
                    "error": item_state.error,
                }
            )
            continue
        result = read_artifact(item_state.result_artifact or "", "rs_agent_result")
        payload = result.payload
        evidence = payload.get("evidence") or {}
        consensus = evidence.get("consensus", "unknown")
        consensus_counts[consensus] += 1
        has_conflict = bool(evidence.get("has_conflict"))
        conflict_items += int(has_conflict)

        original_stance = next(
            (
                claim.get("stance")
                for claim in evidence.get("claims", [])
                if claim.get("kind") == "original_caption"
            ),
            None,
        )
        mask_stance = next(
            (
                claim.get("stance")
                for claim in evidence.get("claims", [])
                if claim.get("kind") == "mask"
            ),
            None,
        )
        if original_stance and mask_stance:
            mask_caption_counts[
                "caption_{}_mask_{}".format(original_stance, mask_stance)
            ] += 1

        item_rows.append(
            {
                "item_id": item_id,
                "run_id": item_state.run_id,
                "task_type": payload.get("task_type"),
                "original_caption": payload.get("original_caption"),
                "selected_caption": payload.get("selected_caption"),
                "selected_answers": payload.get("selected_answers", []),
                "mask": payload.get("mask"),
                "consensus": consensus,
                "has_conflict": has_conflict,
                "conflict_count": len(evidence.get("conflicts", [])),
                "result_artifact": item_state.result_artifact,
            }
        )
        item_caption_rows, item_caption_requests = _caption_data(item_id, payload)
        item_vqa_rows, item_vqa_requests = _vqa_data(item_id, payload)
        caption_rows.extend(item_caption_rows)
        vqa_rows.extend(item_vqa_rows)
        request_rows.extend(item_caption_requests)
        request_rows.extend(item_vqa_requests)

    model_rows = _model_summary(request_rows, caption_rows, vqa_rows)
    metric_rows, metric_records, missing_references, references_sha256 = (
        _caption_metric_rows(item_rows, references_path)
    )
    summary = {
        "batch_id": state.batch_id,
        "batch_status": state.status.value,
        "experiment_fingerprint": state.experiment.fingerprint,
        "state_counts": state.counts(),
        "exported_item_count": len(item_rows),
        "failure_count": len(failure_rows),
        "conflict_item_count": conflict_items,
        "consensus_counts": dict(consensus_counts),
        "caption_mask_counts": dict(mask_caption_counts),
        "caption_candidate_rows": len(caption_rows),
        "caption_generation_failures": sum(
            not bool(row["success"]) for row in caption_rows
        ),
        "vqa_candidate_rows": len(vqa_rows),
        "vqa_generation_failures": sum(
            not bool(row["success"]) for row in vqa_rows
        ),
        "request_telemetry_rows": len(request_rows),
        "caption_reference_metrics": mean_caption_metrics(metric_records),
        "caption_reference_matched_items": len(metric_records),
        "caption_reference_missing_items": missing_references,
        "caption_references_sha256": references_sha256,
        "caption_reference_metrics_version": REFERENCE_METRICS_VERSION,
        "export_source_sha256": source_tree_sha256(),
        "metric_definitions": {
            "primary": "LLM-as-Judge score and highest_score_then_judge selection",
            "bleu": "dependency-light unsmoothed sentence BLEU averaged over items",
            "rouge_l": "best-reference ROUGE-L F-score with beta=1.2",
            "change_flag": "keyword-derived selected-caption flag versus LEVIR-MCI annotation",
            "latency": "end-to-end provider request wall time including retries",
            "score_std": "population standard deviation",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "experiment_identity.json", state.experiment.model_dump(mode="json")
    )
    _write_jsonl(output_dir / "items.jsonl", item_rows)
    _write_jsonl(output_dir / "failures.jsonl", failure_rows)
    common = [
        "item_id",
        "label",
        "model_name",
        "provider",
        "model_id",
        "success",
        "error",
        "score",
        "selected",
        "judge_choice",
        "attempts",
        "latency_ms",
        "status_code",
        "attempt_status_codes",
        "attempt_outcomes",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "text",
    ]
    _write_csv(
        output_dir / "caption_scores.csv",
        caption_rows,
        [common[0], "input_mode", *common[1:]],
    )
    _write_csv(
        output_dir / "vqa_scores.csv",
        vqa_rows,
        ["item_id", "question_id", "question", "question_source", *common[1:]],
    )
    request_columns = [
        "item_id",
        "stage",
        "role",
        "question_id",
        "label",
        "input_mode",
        "model_name",
        "provider",
        "model_id",
        "success",
        "error",
        "attempts",
        "latency_ms",
        "status_code",
        "attempt_status_codes",
        "attempt_outcomes",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    _write_csv(output_dir / "request_telemetry.csv", request_rows, request_columns)
    _write_csv(
        output_dir / "model_summary.csv",
        model_rows,
        [
            "stage",
            "role",
            "model_name",
            "provider",
            "model_id",
            "input_mode",
            "request_count",
            "successful_count",
            "failure_count",
            "success_rate",
            "score_count",
            "mean_score",
            "population_std",
            "selected_count",
            "judge_choice_count",
            "attempt_count",
            "retry_count",
            "mean_latency_ms",
            "p95_latency_ms",
            "known_usage_count",
            "total_tokens",
        ],
    )
    _write_csv(
        output_dir / "caption_reference_metrics.csv",
        metric_rows,
        [
            "item_id",
            "split",
            "reference_count",
            "change_flag",
            "selected_caption",
            "bleu_1",
            "bleu_4",
            "rouge_l",
            "change_flag_match",
        ],
    )
    return summary
