import csv
import json
from pathlib import Path

import pytest

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.evaluation.batch_export import export_batch
from rs_agent.experiments.identity import ExperimentIdentity
from rs_agent.experiments.state import (
    BatchItemState,
    BatchRunState,
    BatchStatus,
    ItemStatus,
)


def write_artifact(
    root: Path, artifact_type: str, run_id: str, item_id: str, payload: dict
) -> str:
    artifact = ArtifactEnvelope.create(
        artifact_type=artifact_type,
        run_id=run_id,
        item_id=item_id,
        payload=payload,
    )
    return str(JsonArtifactStore(root).write(artifact))


def build_state(tmp_path: Path) -> BatchRunState:
    run_id = "export-item-1"
    item_id = "item-1"
    root = tmp_path / "artifacts"
    candidate = {
        "candidate_id": "candidate-a",
        "label": "A",
        "model": {"name": "Model A", "provider": "fake", "model": "vendor/a"},
        "text": "No change is visible.",
        "response": None,
        "error": None,
    }
    failed = {
        "candidate_id": "candidate-b",
        "label": "B",
        "model": {"name": "Model B", "provider": "fake", "model": "vendor/b"},
        "text": "",
        "response": None,
        "error": "ProviderError: upstream failure",
    }
    generation = write_artifact(
        root,
        "caption_candidates",
        run_id,
        item_id,
        {"candidates": [candidate, failed]},
    )
    caption_evaluation = write_artifact(
        root,
        "caption_evaluation",
        run_id,
        item_id,
        {
            "original_caption": "No change.",
            "candidates": [candidate],
            "judge_choice": "A",
            "scores": {"A": 9},
            "selection": {"selected_label": "A"},
        },
    )
    caption_result = write_artifact(
        root,
        "rs_cc_result",
        run_id,
        item_id,
        {
            "source_artifacts": {
                "generation": generation,
                "evaluation": caption_evaluation,
            }
        },
    )
    final = write_artifact(
        root,
        "rs_agent_result",
        run_id,
        item_id,
        {
            "item_id": item_id,
            "task_type": "caption_enrichment",
            "original_caption": "No change.",
            "selected_caption": "No change is visible.",
            "selected_answers": [],
            "mask": None,
            "evidence": {
                "claims": [],
                "conflicts": [],
                "consensus": "no_change",
                "has_conflict": False,
            },
            "source_artifacts": {"rs_cc_result": caption_result, "rs_vqa_result": None},
        },
    )
    return BatchRunState(
        batch_id="export-test",
        experiment=ExperimentIdentity(
            fingerprint="f" * 64,
            input_sha256="i" * 64,
            source_sha256="s" * 64,
            config_sha256={"cc": "c" * 64},
            options={"task": "caption"},
            item_fingerprints={item_id: "x" * 64},
        ),
        status=BatchStatus.COMPLETED,
        items={
            item_id: BatchItemState(
                item_id=item_id,
                item_fingerprint="x" * 64,
                status=ItemStatus.COMPLETED,
                attempts=1,
                run_id=run_id,
                result_artifact=final,
            )
        },
    )


def test_export_keeps_failed_candidates_and_writes_model_summary(tmp_path: Path) -> None:
    state = build_state(tmp_path)
    output = tmp_path / "export"

    summary = export_batch(state, output)

    assert summary["exported_item_count"] == 1
    assert summary["caption_candidate_rows"] == 2
    assert summary["caption_generation_failures"] == 1
    with (output / "model_summary.csv").open("r", encoding="utf-8") as handle:
        rows = {row["model_name"]: row for row in csv.DictReader(handle)}
    assert rows["Model A"]["mean_score"] == "9.0"
    assert rows["Model A"]["successful_count"] == "1"
    assert rows["Model B"]["failure_count"] == "1"
    assert rows["Model B"]["mean_score"] == ""
    with (output / "caption_scores.csv").open("r", encoding="utf-8") as handle:
        candidate_rows = {row["model_name"]: row for row in csv.DictReader(handle)}
    assert candidate_rows["Model B"]["success"] == "False"
    assert "upstream failure" in candidate_rows["Model B"]["error"]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))[
        "failure_count"
    ] == 0


def test_export_rejects_tampered_result_artifact(tmp_path: Path) -> None:
    state = build_state(tmp_path)
    result_path = Path(state.items["item-1"].result_artifact or "")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["payload"]["selected_caption"] = "tampered"
    result_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        export_batch(state, tmp_path / "export")
