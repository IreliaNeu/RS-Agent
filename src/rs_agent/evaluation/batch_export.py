"""Checksum-verified exports from resumable batch state and immutable artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
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


def _caption_rows(
    item_id: str, result_payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    result_path = result_payload["source_artifacts"].get("rs_cc_result")
    if not result_path:
        return []
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
    return [
        {
            "item_id": item_id,
            "label": candidate["label"],
            "model_name": candidate["model"]["name"],
            "provider": candidate["model"]["provider"],
            "model_id": candidate["model"]["model"],
            "success": not bool(candidate.get("error")),
            "error": candidate.get("error"),
            "score": scores.get(candidate["label"]),
            "selected": candidate["label"] == selected,
            "judge_choice": candidate["label"] == judge_choice,
            "text": candidate.get("text", ""),
        }
        for candidate in generation["candidates"]
    ]


def _vqa_rows(item_id: str, result_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    result_path = result_payload["source_artifacts"].get("rs_vqa_result")
    if not result_path:
        return []
    vqa_result = read_artifact(result_path, "rs_vqa_result")
    source_artifacts = vqa_result.payload["source_artifacts"]
    generation = read_artifact(
        source_artifacts["generation"], "vqa_candidates"
    ).payload
    evaluation = read_artifact(
        source_artifacts["evaluation"], "vqa_evaluation"
    ).payload
    evaluation_by_question = {
        record["question"]["question_id"]: record
        for record in evaluation["records"]
    }
    questions = {
        question["question_id"]: question for question in generation["questions"]
    }
    rows = []
    for candidate in generation["candidates"]:
        question_id = candidate["question_id"]
        question = questions[question_id]
        record = evaluation_by_question.get(question_id)
        scores = record.get("scores", {}) if record else {}
        selected = record["selection"]["selected_label"] if record else None
        judge_choice = (record.get("judge_choice") or None) if record else None
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
                "text": candidate.get("text", ""),
            }
        )
    return rows


def _model_summary(
    caption_rows: List[Dict[str, Any]], vqa_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for stage, rows in (("rs_cc", caption_rows), ("rs_vqa", vqa_rows)):
        for row in rows:
            grouped[
                (stage, row["model_name"], row["provider"], row["model_id"])
            ].append(row)
    output = []
    for (stage, name, provider, model_id), rows in sorted(grouped.items()):
        scores = [float(row["score"]) for row in rows if row["score"] is not None]
        successful = sum(bool(row["success"]) for row in rows)
        output.append(
            {
                "stage": stage,
                "model_name": name,
                "provider": provider,
                "model_id": model_id,
                "candidate_count": len(rows),
                "successful_count": successful,
                "failure_count": len(rows) - successful,
                "score_count": len(scores),
                "mean_score": round(statistics.fmean(scores), 6) if scores else None,
                "population_std": (
                    round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0
                )
                if scores
                else None,
                "selected_count": sum(bool(row["selected"]) for row in rows),
                "judge_choice_count": sum(bool(row["judge_choice"]) for row in rows),
            }
        )
    return output


def export_batch(state: BatchRunState, output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    item_rows = []
    failure_rows = []
    caption_rows: List[Dict[str, Any]] = []
    vqa_rows: List[Dict[str, Any]] = []
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
        caption_rows.extend(_caption_rows(item_id, payload))
        vqa_rows.extend(_vqa_rows(item_id, payload))

    model_rows = _model_summary(caption_rows, vqa_rows)
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
        "vqa_generation_failures": sum(not bool(row["success"]) for row in vqa_rows),
        "score_std_definition": "population standard deviation",
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
        "text",
    ]
    _write_csv(output_dir / "caption_scores.csv", caption_rows, common)
    _write_csv(
        output_dir / "vqa_scores.csv",
        vqa_rows,
        [
            "item_id",
            "question_id",
            "question",
            "question_source",
            *common[1:],
        ],
    )
    _write_csv(
        output_dir / "model_summary.csv",
        model_rows,
        [
            "stage",
            "model_name",
            "provider",
            "model_id",
            "candidate_count",
            "successful_count",
            "failure_count",
            "score_count",
            "mean_score",
            "population_std",
            "selected_count",
            "judge_choice_count",
        ],
    )
    return summary
