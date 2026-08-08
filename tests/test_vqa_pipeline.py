import asyncio
import json
from pathlib import Path

from PIL import Image

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ImagePair, ModelResponse
from rs_agent.domains.remote_sensing.vqa_agent import RSVQARequest
from rs_agent.domains.remote_sensing.vqa_config import RSVQAExperimentConfig
from rs_agent.orchestration.vqa_pipeline import RSVQAPipeline


def experiment_config() -> RSVQAExperimentConfig:
    return RSVQAExperimentConfig.model_validate(
        {
            "profile": "test_vqa",
            "providers": {
                "fake": {
                    "base_url": "https://example.test/v1",
                    "api_key_env": "UNUSED_TEST_KEY",
                    "max_concurrency": 5,
                }
            },
            "answer_models": [
                {
                    "name": "generator-{}".format(label),
                    "provider": "fake",
                    "model": "v-{}".format(label.lower()),
                    "request_options": (
                        {"reasoning": {"enabled": True}} if label == "A" else {}
                    ),
                }
                for label in "ABCDE"
            ],
            "selector": {"name": "selector", "provider": "fake", "model": "judge"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge"},
            "default_template_ids": ["change_summary"],
            "minimum_successful_answers_per_question": 5,
        }
    )


class FakeProvider:
    def __init__(self):
        self.calls = []

    async def complete(self, messages, model, temperature, max_tokens, metadata=None):
        self.calls.append(
            {"messages": messages, "model": model, "metadata": metadata}
        )
        if model.startswith("v-"):
            return ModelResponse(
                provider="fake",
                model=model,
                content="Visual answer from {}".format(model),
            )
        if "score" in messages[0]["content"].lower():
            return ModelResponse(
                provider="fake",
                model=model,
                content='{"A": 7, "B": 10, "C": 8, "D": 6, "E": 9}',
            )
        return ModelResponse(provider="fake", model=model, content="A")


class FakeRegistry:
    def __init__(self, provider: FakeProvider):
        self.provider = provider

    def get(self, name: str) -> FakeProvider:
        assert name == "fake"
        return self.provider


def make_pair(root: Path) -> ImagePair:
    before = root / "before.png"
    after = root / "after.png"
    Image.new("RGB", (4, 4), color=(20, 30, 40)).save(before)
    Image.new("RGB", (4, 4), color=(40, 30, 20)).save(after)
    return ImagePair(before=before, after=after)


def test_vqa_pipeline_uses_original_pair_selects_answer_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    pipeline = RSVQAPipeline(
        experiment_config(), FakeRegistry(provider), JsonArtifactStore(tmp_path / "artifacts")
    )
    result = asyncio.run(
        pipeline.run(
            RSVQARequest(item_id="pair-1", images=make_pair(tmp_path)),
            run_id="vqa-test",
        )
    )
    assert result.question_count == 1
    assert result.selected_answers[0].selection.selected_label == "B"
    assert result.selected_answers[0].answer == "Visual answer from v-b"
    assert len(list((tmp_path / "artifacts").rglob("*.json"))) == 3

    generation_calls = [call for call in provider.calls if call["model"].startswith("v-")]
    assert len(generation_calls) == 5
    image_parts = [
        part
        for part in generation_calls[0]["messages"][1]["content"]
        if part["type"] == "image_url"
    ]
    assert len(image_parts) == 2
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in image_parts)
    assert generation_calls[0]["metadata"] == {"reasoning": {"enabled": True}}
    assert "mask" not in json.dumps(generation_calls[0]["messages"]).lower()

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "artifacts").rglob("*.json")
    )
    assert "data:image" not in artifact_text
