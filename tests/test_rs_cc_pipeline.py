import asyncio
import json
from pathlib import Path

import pytest

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ModelResponse
from rs_agent.domains.remote_sensing.caption_agent import RSCCAgent, RSCCRequest
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.orchestration.caption_pipeline import (
    InsufficientCandidatesError,
    RSCCPipeline,
)


def experiment_config(minimum: int = 5) -> RSCCExperimentConfig:
    return RSCCExperimentConfig.model_validate(
        {
            "profile": "test_rs_cc",
            "providers": {
                "fake": {
                    "base_url": "https://example.test/v1",
                    "api_key_env": "UNUSED_TEST_KEY",
                    "max_concurrency": 5,
                }
            },
            "caption_generators": [
                {"name": "generator-{}".format(label), "provider": "fake", "model": "g-{}".format(label.lower())}
                for label in "ABCDE"
            ],
            "selector": {"name": "selector", "provider": "fake", "model": "judge"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge"},
            "minimum_successful_candidates": minimum,
        }
    )


class FakeProvider:
    def __init__(self, failing_model: str = ""):
        self.failing_model = failing_model
        self.active = 0
        self.max_active = 0

    async def complete(self, messages, model, temperature, max_tokens, metadata=None):
        if model.startswith("g-"):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            if model == self.failing_model:
                raise RuntimeError("planned model failure")
            return ModelResponse(
                provider="fake",
                model=model,
                content="Caption from {}".format(model),
                raw_response={"id": model, "test": True},
            )
        if "Score each" in messages[0]["content"]:
            return ModelResponse(
                provider="fake",
                model=model,
                content='{"A": 7, "B": 10, "C": 8, "D": 6, "E": 9}',
                raw_response={"kind": "scores"},
            )
        return ModelResponse(
            provider="fake",
            model=model,
            content="A",
            raw_response={"kind": "choice"},
        )


class FakeRegistry:
    def __init__(self, provider: FakeProvider):
        self.provider = provider

    def get(self, name: str) -> FakeProvider:
        assert name == "fake"
        return self.provider


def test_five_candidate_calls_run_concurrently() -> None:
    provider = FakeProvider()
    agent = RSCCAgent(experiment_config(), FakeRegistry(provider))
    batch = asyncio.run(
        agent.generate(RSCCRequest(item_id="test_1", original_caption="One building appeared."))
    )
    assert [candidate.label for candidate in batch.candidates] == list("ABCDE")
    assert provider.max_active == 5
    assert not batch.errors


def test_pipeline_selects_highest_score_and_writes_three_artifacts(tmp_path: Path) -> None:
    config = experiment_config()
    registry = FakeRegistry(FakeProvider())
    store = JsonArtifactStore(tmp_path)
    pipeline = RSCCPipeline(config, registry, store)
    result = asyncio.run(
        pipeline.run(
            RSCCRequest(item_id="test_1", original_caption="One building appeared."),
            run_id="paper-test",
        )
    )
    assert result.selected_label == "B"
    assert result.selected_caption == "Caption from g-b"
    paths = list(tmp_path.rglob("*.json"))
    assert len(paths) == 3
    generation_path = next(path for path in paths if "caption_candidates" in path.parts)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    assert generation["payload"]["candidates"][0]["response"]["raw_response"]["test"] is True


def test_pipeline_preserves_generation_artifact_before_strict_failure(tmp_path: Path) -> None:
    config = experiment_config(minimum=5)
    pipeline = RSCCPipeline(
        config,
        FakeRegistry(FakeProvider(failing_model="g-e")),
        JsonArtifactStore(tmp_path),
    )
    with pytest.raises(InsufficientCandidatesError) as error:
        asyncio.run(
            pipeline.run(
                RSCCRequest(item_id="test_2", original_caption="A road disappeared."),
                run_id="failure-test",
            )
        )
    assert error.value.generation_artifact.exists()
    assert len(list(tmp_path.rglob("*.json"))) == 1
