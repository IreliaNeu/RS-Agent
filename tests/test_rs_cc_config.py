from pathlib import Path

import pytest
from pydantic import ValidationError

from rs_agent.domains.remote_sensing.config import load_rs_cc_config


def test_paper_rs_cc_config_has_exact_model_order() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_rs_cc_config(root / "configs" / "rs_cc.paper.yaml")
    assert [model.model for model in config.caption_generators] == [
        "anthropic/claude-sonnet-4",
        "deepseek/deepseek-chat-v3-0324",
        "openai/gpt-4o-mini",
        "qwen/qwen3-30b-a3b",
        "meta-llama/llama-4-maverick",
    ]
    assert config.selector.model == "openai/gpt-4o"
    assert config.evaluator.model == "openai/gpt-4o"


def test_paper_rs_cc_config_rejects_duplicate_candidate_models(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "rs_cc.paper.yaml").read_text(encoding="utf-8")
    text = text.replace("deepseek/deepseek-chat-v3-0324", "anthropic/claude-sonnet-4")
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_rs_cc_config(path)
