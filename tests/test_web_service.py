import asyncio
from pathlib import Path

from PIL import Image

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ImagePair, ModelResponse
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.domains.remote_sensing.vqa_config import RSVQAExperimentConfig
from rs_agent.orchestration.agent_pipeline import RSAgentPipeline, RSAgentRequest
from rs_agent.web.service import build_demo_view


def provider_config() -> dict:
    return {
        "fake": {
            "base_url": "https://example.test/v1",
            "api_key_env": "UNUSED_TEST_KEY",
        }
    }


def models(prefix: str) -> list:
    return [
        {
            "name": "{}-{}".format(prefix, label),
            "provider": "fake",
            "model": "{}-{}".format(prefix, label.lower()),
        }
        for label in "ABCDE"
    ]


class FakeProvider:
    async def complete(self, messages, model, temperature, max_tokens, metadata=None):
        if model.startswith("cc-"):
            return ModelResponse(provider="fake", model=model, content="caption {}".format(model))
        if model.startswith("vqa-"):
            return ModelResponse(provider="fake", model=model, content="answer {}".format(model))
        if "score" in str(messages).lower():
            return ModelResponse(
                provider="fake",
                model=model,
                content='{"A": 7, "B": 10, "C": 8, "D": 6, "E": 9}',
            )
        return ModelResponse(provider="fake", model=model, content="B")


class FakeRegistry:
    def __init__(self):
        self.provider = FakeProvider()

    def get(self, name):
        return self.provider


def test_demo_view_contains_candidate_ledgers_and_selected_outputs(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(before)
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(after)
    cc_config = RSCCExperimentConfig.model_validate(
        {
            "profile": "web-cc-test",
            "providers": provider_config(),
            "caption_generators": models("cc"),
            "selector": {"name": "selector", "provider": "fake", "model": "judge"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge"},
        }
    )
    vqa_config = RSVQAExperimentConfig.model_validate(
        {
            "profile": "web-vqa-test",
            "providers": provider_config(),
            "answer_models": models("vqa"),
            "selector": {"name": "selector", "provider": "fake", "model": "judge"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge"},
            "default_template_ids": ["change_summary"],
        }
    )
    registry = FakeRegistry()
    result = asyncio.run(
        RSAgentPipeline(
            cc_config,
            registry,
            vqa_config,
            registry,
            JsonArtifactStore(tmp_path / "artifacts"),
        ).run(
            RSAgentRequest(
                item_id="pair-1",
                original_caption="A building appeared.",
                images=ImagePair(before=before, after=after),
            ),
            "web-test",
        )
    )

    view = build_demo_view(result)

    assert len(view.caption_candidates) == 5
    assert len(view.vqa_candidates) == 5
    assert view.selected_caption == "caption cc-b"
    assert view.selected_answers[0].answer == "answer vqa-b"
    assert view.knowledge_caption == "caption cc-b"
    assert set(view.artifacts) >= {"rs_agent_result", "rs_cc_result", "rs_vqa_result"}
