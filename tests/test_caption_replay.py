import asyncio
from pathlib import Path

import pytest

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.core.schemas import CaptionCandidate, ModelRef, ModelResponse
from rs_agent.domains.remote_sensing.caption_agent import (
    CaptionReplayCandidate,
    RSCCAgent,
    RSCCRequest,
)
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.experiments.identity import ExperimentIdentity
from rs_agent.experiments.replay import load_caption_replays
from rs_agent.experiments.state import (
    BatchItemState,
    BatchRunState,
    BatchStateStore,
    BatchStatus,
    ItemStatus,
)


def config() -> RSCCExperimentConfig:
    return RSCCExperimentConfig.model_validate(
        {
            "profile": "replay-test",
            "providers": {
                "fake": {
                    "base_url": "https://example.test/v1",
                    "api_key_env": "UNUSED_TEST_KEY",
                }
            },
            "caption_generators": [
                {
                    "name": "generator-{}".format(label),
                    "provider": "fake",
                    "model": "g-{}".format(label.lower()),
                }
                for label in "ABCDE"
            ],
            "selector": {"name": "selector", "provider": "fake", "model": "judge"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge"},
        }
    )


class FakeProvider:
    def __init__(self):
        self.models = []

    async def complete(self, messages, model, temperature, max_tokens, metadata=None):
        self.models.append(model)
        return ModelResponse(provider="fake", model=model, content="new {}".format(model))


class FakeRegistry:
    def __init__(self, provider):
        self.provider = provider

    def get(self, name):
        return self.provider


def replay(label: str) -> CaptionReplayCandidate:
    return CaptionReplayCandidate(
        label=label,
        model=ModelRef(
            name="generator-{}".format(label),
            provider="fake",
            model="g-{}".format(label.lower()),
        ),
        input_mode="text_only",
        text="frozen {}".format(label),
        source_candidate_id="source-{}".format(label),
        source_artifact="/source/generation.json",
    )


def test_agent_replays_selected_slots_without_provider_requests() -> None:
    provider = FakeProvider()
    agent = RSCCAgent(config(), FakeRegistry(provider))
    result = asyncio.run(
        agent.generate(
            RSCCRequest(
                item_id="item-1",
                original_caption="No change.",
                replayed_candidates={"A": replay("A"), "B": replay("B")},
            )
        )
    )

    assert provider.models == ["g-c", "g-d", "g-e"]
    assert [candidate.generation_mode for candidate in result.candidates] == [
        "replayed",
        "replayed",
        "generated",
        "generated",
        "generated",
    ]
    assert result.candidates[0].response is None
    assert result.candidates[0].source_candidate_id == "source-A"


def test_agent_rejects_replay_from_a_different_model() -> None:
    candidate = replay("A").model_copy(
        update={"model": ModelRef(name="wrong", provider="fake", model="g-a")}
    )
    with pytest.raises(ValueError, match="model identity"):
        asyncio.run(
            RSCCAgent(config(), FakeRegistry(FakeProvider())).generate(
                RSCCRequest(
                    item_id="item-1",
                    original_caption="No change.",
                    replayed_candidates={"A": candidate},
                )
            )
        )


def test_loader_reads_checksum_verified_completed_artifacts(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path / "artifacts")
    candidate = CaptionCandidate(
        candidate_id="candidate-a",
        label="A",
        model=ModelRef(name="generator-A", provider="fake", model="g-a"),
        text="Frozen caption.",
    )
    generation = store.write(
        ArtifactEnvelope.create(
            artifact_type="caption_candidates",
            run_id="source-run",
            item_id="item-1",
            payload={"candidates": [candidate.model_dump(mode="json")]},
        )
    )
    caption_result = store.write(
        ArtifactEnvelope.create(
            artifact_type="rs_cc_result",
            run_id="source-run",
            item_id="item-1",
            payload={"source_artifacts": {"generation": str(generation)}},
        )
    )
    agent_result = store.write(
        ArtifactEnvelope.create(
            artifact_type="rs_agent_result",
            run_id="source-run",
            item_id="item-1",
            payload={"source_artifacts": {"rs_cc_result": str(caption_result)}},
        )
    )
    identity = ExperimentIdentity(
        fingerprint="f",
        input_sha256="i",
        source_sha256="s",
        runtime={},
        config_sha256={},
        options={},
        item_fingerprints={"item-1": "item"},
    )
    state_path = tmp_path / "state.json"
    BatchStateStore(state_path).write(
        BatchRunState(
            batch_id="source",
            experiment=identity,
            status=BatchStatus.COMPLETED,
            items={
                "item-1": BatchItemState(
                    item_id="item-1",
                    item_fingerprint="item",
                    status=ItemStatus.COMPLETED,
                    attempts=1,
                    run_id="source-run",
                    result_artifact=str(agent_result),
                )
            },
        )
    )

    loaded = load_caption_replays(state_path, ["A"])

    assert loaded["item-1"]["A"].text == "Frozen caption."
    assert loaded["item-1"]["A"].source_candidate_id == "candidate-a"
    assert loaded["item-1"]["A"].source_artifact == str(generation.resolve())
