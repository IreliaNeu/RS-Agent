from pathlib import Path

from rs_agent.domains.remote_sensing.vqa_config import load_rs_vqa_config


def test_paper_vqa_config_has_the_paper_model_group() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_rs_vqa_config(root / "configs" / "rs_vqa.paper.yaml")
    assert [model.model for model in config.answer_models] == [
        "minimax/minimax-01",
        "openai/gpt-4o-mini",
        "meta-llama/llama-4-maverick",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "mistralai/mistral-small-3.2-24b-instruct",
    ]
    assert config.selector.model == "openai/gpt-4o"
    assert config.evaluator.model == "openai/gpt-4o"
