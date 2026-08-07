import json

import pytest

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore


def test_artifact_store_is_write_once_and_verifies_checksum(tmp_path) -> None:
    store = JsonArtifactStore(tmp_path)
    artifact = ArtifactEnvelope.create(
        artifact_type="caption_candidates",
        run_id="run-1",
        item_id="sample-1",
        payload={"captions": ["one", "two"]},
        artifact_id="artifact-1",
    )

    path = store.write(artifact)
    assert store.read(path) == artifact
    with pytest.raises(FileExistsError):
        store.write(artifact)

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["payload"]["captions"][0] = "tampered"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.read(path)

