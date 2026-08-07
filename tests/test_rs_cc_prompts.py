from rs_agent.domains.remote_sensing.caption_prompts import (
    caption_enrichment_messages,
    clean_caption_output,
)
from rs_agent.domains.remote_sensing.config import CaptionPromptProfile


def test_rs_cc_prompt_is_text_only_and_contains_original_caption() -> None:
    messages = caption_enrichment_messages(
        "One building was constructed.", CaptionPromptProfile.BASE
    )
    assert messages[1]["content"].endswith("One building was constructed.")
    assert all(isinstance(message["content"], str) for message in messages)
    assert "image_url" not in str(messages)


def test_clean_caption_output_handles_json_and_markdown() -> None:
    assert clean_caption_output('{"refined_caption":"A new building appeared."}') == (
        "A new building appeared."
    )
    assert clean_caption_output("```text\nCaption: Two roads disappeared.\n```") == (
        "Two roads disappeared."
    )
