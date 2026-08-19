from pathlib import Path

from rs_agent.domains.remote_sensing.config import (
    CaptionInputMode,
    load_rs_cc_config,
)


def test_paired_ablation_profiles_use_identical_models_and_different_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    text = load_rs_cc_config(root / "configs" / "rs_cc.ablation_text_only.yaml")
    mixed = load_rs_cc_config(root / "configs" / "rs_cc.ablation_mixed.yaml")

    assert [model.model for model in text.caption_generators] == [
        model.model for model in mixed.caption_generators
    ]
    assert {model.input_mode for model in text.caption_generators} == {
        CaptionInputMode.TEXT_ONLY
    }
    assert [model.input_mode for model in mixed.caption_generators].count(
        CaptionInputMode.IMAGE_TEXT
    ) == 3
    assert text.protocol.baseline_id == mixed.protocol.baseline_id
