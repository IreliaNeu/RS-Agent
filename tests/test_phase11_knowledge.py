import json
from pathlib import Path

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.experiments.identity import ExperimentIdentity
from rs_agent.experiments.replay import load_selected_knowledge
from rs_agent.experiments.state import (
    BatchItemState,
    BatchRunState,
    BatchStateStore,
    BatchStatus,
    ItemStatus,
)


def test_load_selected_knowledge_preserves_c_star_provenance(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path / "artifacts")
    caption = store.write(
        ArtifactEnvelope.create(
            artifact_type="rs_cc_result",
            run_id="caption-run",
            item_id="pair-1",
            payload={"selected_caption": "A new building appears."},
        )
    )
    result = store.write(
        ArtifactEnvelope.create(
            artifact_type="rs_agent_result",
            run_id="caption-run",
            item_id="pair-1",
            payload={"source_artifacts": {"rs_cc_result": str(caption)}},
        )
    )
    state_path = tmp_path / "state.json"
    BatchStateStore(state_path).write(
        BatchRunState(
            batch_id="caption-source",
            experiment=ExperimentIdentity(
                fingerprint="f",
                input_sha256="i",
                source_sha256="s",
                runtime={},
                config_sha256={},
                options={},
                item_fingerprints={"pair-1": "x"},
            ),
            status=BatchStatus.COMPLETED,
            items={
                "pair-1": BatchItemState(
                    item_id="pair-1",
                    item_fingerprint="x",
                    status=ItemStatus.COMPLETED,
                    attempts=1,
                    run_id="caption-run",
                    result_artifact=str(result),
                )
            },
        )
    )

    packet = load_selected_knowledge(state_path)["pair-1"]

    assert packet.c_star == "A new building appears."
    assert packet.run_id == "caption-run"
    assert packet.source_result_artifact == str(caption.resolve())
    source = json.loads(caption.read_text(encoding="utf-8"))
    assert source["payload"]["selected_caption"] == packet.c_star
