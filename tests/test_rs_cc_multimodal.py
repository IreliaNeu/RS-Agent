import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from rs_agent.core.schemas import ImagePair, ModelResponse
from rs_agent.domains.remote_sensing.caption_agent import RSCCAgent, RSCCRequest
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig


class RecordingProvider:
    def __init__(self):
        self.calls = []

    async def complete(self, messages, model, temperature, max_tokens, metadata=None):
        self.calls.append({"model": model, "messages": messages})
        return ModelResponse(provider="fake", model=model, content="A grounded caption.")


class Registry:
    def __init__(self, provider):
        self.provider = provider

    def get(self, name):
        assert name == "fake"
        return self.provider


def config(track="enhancement"):
    modes = ["text_only", "text_only", "image_text", "image_text", "image_text"]
    return RSCCExperimentConfig.model_validate(
        {
            "profile": "mixed-test",
            "protocol": {
                "track": track,
                "baseline_id": "paper-v1",
                "method_variant": "mixed",
                "substitutes_paper_models": track != "paper",
            },
            "providers": {
                "fake": {
                    "base_url": "https://example.test/v1",
                    "api_key_env": "UNUSED_TEST_KEY",
                }
            },
            "caption_generators": [
                {
                    "name": "model-{}".format(label),
                    "provider": "fake",
                    "model": "vendor/{}".format(label.lower()),
                    "input_mode": mode,
                }
                for label, mode in zip("ABCDE", modes)
            ],
            "selector": {"name": "selector", "provider": "fake", "model": "judge-a"},
            "evaluator": {"name": "evaluator", "provider": "fake", "model": "judge-b"},
            "minimum_successful_candidates": 4,
        }
    )


def make_pair(root: Path) -> ImagePair:
    before = root / "before.png"
    after = root / "after.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(before)
    Image.new("RGB", (4, 4), color=(30, 20, 10)).save(after)
    return ImagePair(before=before, after=after)


def test_mixed_rs_cc_sends_images_only_to_image_text_models(tmp_path: Path) -> None:
    provider = RecordingProvider()
    batch = asyncio.run(
        RSCCAgent(config(), Registry(provider)).generate(
            RSCCRequest(
                item_id="pilot-1",
                original_caption="A building appeared.",
                images=make_pair(tmp_path),
            )
        )
    )

    by_model = {call["model"]: call["messages"] for call in provider.calls}
    assert "data:image" not in json.dumps(by_model["vendor/a"])
    assert "data:image" not in json.dumps(by_model["vendor/b"])
    for model in ("vendor/c", "vendor/d", "vendor/e"):
        assert json.dumps(by_model[model]).count("data:image") == 2
    serialized = json.dumps(batch.model_dump(mode="json"))
    assert "data:image" not in serialized
    assert set(batch.image_sha256) == {"before", "after"}
    assert batch.input_modes == {
        "A": "text_only",
        "B": "text_only",
        "C": "image_text",
        "D": "image_text",
        "E": "image_text",
    }


def test_paper_track_rejects_image_text_rs_cc() -> None:
    with pytest.raises(ValueError, match="paper RS-CC track must remain text-only"):
        config(track="paper")
