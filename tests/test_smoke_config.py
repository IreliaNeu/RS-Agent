from pathlib import Path

from rs_agent.domains.remote_sensing.config import load_rs_cc_config


def test_operational_smoke_config_is_explicitly_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_rs_cc_config(root / "configs" / "rs_cc.smoke.yaml")
    assert config.profile == "rs_cc_operational_smoke_v1"
    assert {model.provider for model in config.caption_generators} == {
        "openrouter",
        "siliconflow",
    }
    assert config.selector.provider == "siliconflow"
    assert config.evaluator.provider == "siliconflow"
