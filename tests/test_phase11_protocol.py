from pathlib import Path

from rs_agent.domains.remote_sensing.caption_prompts import (
    caption_enrichment_messages,
    prompt_version,
)
from rs_agent.domains.remote_sensing.config import (
    CaptionPromptProfile,
    load_rs_cc_config,
)


def test_paper_experiment_and_method_interpretation_are_separate() -> None:
    experiment = load_rs_cc_config(Path("configs/rs_cc.paper.yaml"))
    method = load_rs_cc_config(Path("configs/rs_cc.method_image_text.yaml"))

    assert experiment.protocol.method_variant == "paper_experiment_text_only"
    assert not experiment.requires_images
    assert method.protocol.method_variant == (
        "paper_method_image_pair_plus_caption_interpretation"
    )
    assert method.requires_images
    assert method.protocol.substitutes_paper_models is True


def test_cot_background_profile_is_versioned_and_hides_reasoning() -> None:
    messages = caption_enrichment_messages(
        "A building appeared.", CaptionPromptProfile.COT_WITH_BACKGROUND
    )

    assert prompt_version(CaptionPromptProfile.COT_WITH_BACKGROUND).endswith(
        "background_v1"
    )
    assert "remote-sensing" in str(messages)
    assert "no reasoning" in str(messages)
