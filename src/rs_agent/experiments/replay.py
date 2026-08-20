"""Integrity-checked caption candidate replay from a completed batch."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ModelRef
from rs_agent.domains.remote_sensing.caption_agent import CaptionReplayCandidate
from rs_agent.experiments.state import BatchStateStore, ItemStatus
from rs_agent.orchestration.knowledge_bridge import KnowledgeBridgePacket


def _read(path: str, expected_type: str):
    artifact = JsonArtifactStore(Path(".")).read(Path(path))
    if artifact.artifact_type != expected_type:
        raise ValueError(
            "expected {} artifact, got {}".format(
                expected_type, artifact.artifact_type
            )
        )
    return artifact


def normalize_replay_labels(labels: Iterable[str]) -> list[str]:
    normalized = [label.strip().upper() for label in labels]
    if not normalized:
        raise ValueError("at least one replay label is required")
    if len(normalized) != len(set(normalized)):
        raise ValueError("replay labels must be unique")
    invalid = [label for label in normalized if label not in set("ABCDE")]
    if invalid:
        raise ValueError("invalid replay labels: {}".format(invalid))
    return normalized


def load_caption_replays(
    state_path: Path,
    labels: Iterable[str],
) -> Dict[str, Dict[str, CaptionReplayCandidate]]:
    """Load selected slots while preserving their immutable source provenance."""
    selected_labels = normalize_replay_labels(labels)
    state = BatchStateStore(state_path.resolve()).read()
    output: Dict[str, Dict[str, CaptionReplayCandidate]] = {}
    for item_id, item_state in state.items.items():
        if item_state.status != ItemStatus.COMPLETED:
            continue
        if not item_state.result_artifact:
            raise ValueError(
                "completed replay source item {} has no result artifact".format(item_id)
            )
        result = _read(item_state.result_artifact, "rs_agent_result")
        if result.item_id != item_id:
            raise ValueError("replay result item ID does not match state")
        caption_result_path = result.payload.get("source_artifacts", {}).get(
            "rs_cc_result"
        )
        if not caption_result_path:
            raise ValueError("replay source item {} has no RS-CC result".format(item_id))
        caption_result = _read(caption_result_path, "rs_cc_result")
        generation_path = caption_result.payload.get("source_artifacts", {}).get(
            "generation"
        )
        if not generation_path:
            raise ValueError(
                "replay source item {} has no generation artifact".format(item_id)
            )
        generation = _read(generation_path, "caption_candidates")
        candidates = {
            candidate["label"]: candidate
            for candidate in generation.payload.get("candidates", [])
        }
        item_replays = {}
        for label in selected_labels:
            candidate = candidates.get(label)
            if candidate is None:
                raise ValueError(
                    "replay source item {} has no candidate {}".format(item_id, label)
                )
            if candidate.get("error") or not str(candidate.get("text", "")).strip():
                raise ValueError(
                    "replay source item {} candidate {} is unsuccessful".format(
                        item_id, label
                    )
                )
            item_replays[label] = CaptionReplayCandidate(
                label=label,
                model=ModelRef.model_validate(candidate["model"]),
                input_mode=candidate.get("input_mode", "text_only"),
                text=candidate["text"],
                source_candidate_id=candidate["candidate_id"],
                source_artifact=str(Path(generation_path).resolve()),
            )
        output[item_id] = item_replays
    return output

def load_selected_knowledge(
    state_path: Path,
) -> Dict[str, KnowledgeBridgePacket]:
    """Load each completed run's selected C* with immutable artifact provenance."""
    state = BatchStateStore(state_path.resolve()).read()
    output: Dict[str, KnowledgeBridgePacket] = {}
    for item_id, item_state in state.items.items():
        if item_state.status != ItemStatus.COMPLETED:
            continue
        if not item_state.result_artifact:
            raise ValueError(
                "completed knowledge source item {} has no result artifact".format(
                    item_id
                )
            )
        result = _read(item_state.result_artifact, "rs_agent_result")
        if result.item_id != item_id:
            raise ValueError("knowledge result item ID does not match state")
        caption_result_path = result.payload.get("source_artifacts", {}).get(
            "rs_cc_result"
        )
        if not caption_result_path:
            raise ValueError(
                "knowledge source item {} has no RS-CC result".format(item_id)
            )
        caption_result = _read(caption_result_path, "rs_cc_result")
        if caption_result.item_id != item_id:
            raise ValueError("knowledge RS-CC item ID does not match state")
        c_star = str(caption_result.payload.get("selected_caption") or "").strip()
        if not c_star:
            raise ValueError(
                "knowledge source item {} has no selected C*".format(item_id)
            )
        output[item_id] = KnowledgeBridgePacket(
            run_id=caption_result.run_id,
            item_id=item_id,
            c_star=c_star,
            source_result_artifact=str(Path(caption_result_path).resolve()),
        )
    return output
