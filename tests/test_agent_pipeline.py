import asyncio
import json
from pathlib import Path

from PIL import Image

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ImagePair, ModelResponse
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.domains.remote_sensing.vqa_config import RSVQAExperimentConfig
from rs_agent.orchestration.agent_pipeline import RSAgentPipeline, RSAgentRequest


def provider_config() -> dict:
    return {
        "fake": {
            "base_url": "https://example.test/v1",
            "api_key_env": "UNUSED_TEST_KEY",
            "max_concurrency": 5,
        }
    }


def model_group(prefix: str) -> list:
    return [
        {
            "name": "{}-{}".format(prefix, label),
            "provider": "fake",
            "model": "{}-{}".format(prefix, label.lower()),
        }
        for label in "ABCDE"
    ]


def cc_config() -> RSCCExperimentConfig:
    return RSCCExperimentConfig.model_validate(
        {
            "profile": "cc-test",
            "providers": provider_config(),
            "caption_generators": model_group("cc"),
            "selector": {"name": "selector", "provider": "fake", "model": "judge"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge"},
        }
    )


def vqa_config() -> RSVQAExperimentConfig:
    return RSVQAExperimentConfig.model_validate(
        {
            "profile": "vqa-test",
            "providers": provider_config(),
            "answer_models": model_group("vqa"),
            "selector": {"name": "selector", "provider": "fake", "model": "judge"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge"},
            "default_template_ids": ["change_summary"],
        }
    )


class FakeProvider:
    def __init__(self):
        self.calls = []

    async def complete(self, messages, model, temperature, max_tokens, metadata=None):
        self.calls.append({"messages": messages, "model": model})
        if model.startswith("cc-"):
            return ModelResponse(
                provider="fake", model=model, content="CC caption from {}".format(model)
            )
        if model.startswith("vqa-"):
            return ModelResponse(
                provider="fake", model=model, content="VQA answer from {}".format(model)
            )
        if "score" in messages[0]["content"].lower():
            return ModelResponse(
                provider="fake",
                model=model,
                content='{"A": 7, "B": 10, "C": 8, "D": 6, "E": 9}',
            )
        return ModelResponse(provider="fake", model=model, content="B")


class FakeRegistry:
    def __init__(self, provider: FakeProvider):
        self.provider = provider

    def get(self, name: str) -> FakeProvider:
        assert name == "fake"
        return self.provider


def test_complete_pipeline_bridges_only_selected_c_star(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(before)
    Image.new("RGB", (4, 4), color=(30, 20, 10)).save(after)
    provider = FakeProvider()
    registry = FakeRegistry(provider)
    pipeline = RSAgentPipeline(
        cc_config(),
        registry,
        vqa_config(),
        registry,
        JsonArtifactStore(tmp_path / "artifacts"),
    )
    result = asyncio.run(
        pipeline.run(
            RSAgentRequest(
                item_id="pair-1",
                original_caption="A building appeared.",
                images=ImagePair(before=before, after=after),
            ),
            "complete-test",
        )
    )
    assert result.knowledge.c_star == "CC caption from cc-b"
    assert result.vqa.selected_answers[0].answer == "VQA answer from vqa-b"
    assert len(list((tmp_path / "artifacts").rglob("*.json"))) == 7

    vqa_call = next(call for call in provider.calls if call["model"] == "vqa-a")
    prompt = json.dumps(vqa_call["messages"])
    assert "CC caption from cc-b" in prompt
    assert "CC caption from cc-a" not in prompt
    assert prompt.count("data:image/png;base64,") == 2
