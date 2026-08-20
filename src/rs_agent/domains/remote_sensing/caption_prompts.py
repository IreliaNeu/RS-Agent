"""Versioned prompts for text-only and image-grounded RS-CC enrichment."""

from __future__ import annotations

import json
import re
from typing import Dict, List

from rs_agent.domains.remote_sensing.config import CaptionPromptProfile
from rs_agent.domains.remote_sensing.image_inputs import EncodedImagePair
from rs_agent.providers.base import ChatMessage

BASE_PROMPT_VERSION = "rs_cc_enrichment_base_v1"
COT_PROMPT_VERSION = "rs_cc_enrichment_cot_no_background_v1"
COT_BACKGROUND_PROMPT_VERSION = "rs_cc_enrichment_cot_background_v1"
IMAGE_TEXT_PROMPT_VERSION = "rs_cc_enrichment_image_text_v1"


def prompt_version(profile: CaptionPromptProfile) -> str:
    if profile == CaptionPromptProfile.COT_WITHOUT_BACKGROUND:
        return COT_PROMPT_VERSION
    if profile == CaptionPromptProfile.COT_WITH_BACKGROUND:
        return COT_BACKGROUND_PROMPT_VERSION
    return BASE_PROMPT_VERSION


def caption_enrichment_messages(
    original_caption: str, profile: CaptionPromptProfile
) -> List[ChatMessage]:
    if not original_caption.strip():
        raise ValueError("original caption cannot be empty")
    if profile == CaptionPromptProfile.COT_WITHOUT_BACKGROUND:
        instruction = (
            "Analyze the wording internally, then produce one clearer and more informative remote "
            "sensing change caption. Do not add background scenarios or facts that are absent from "
            "the original description. Return only the final caption and no reasoning."
        )
    elif profile == CaptionPromptProfile.COT_WITH_BACKGROUND:
        instruction = (
            "Reason internally using remote-sensing change-detection knowledge about temporal "
            "ordering, land-cover objects, spatial relations, and construction or removal. Use "
            "that knowledge to improve terminology and organization, but never assert a concrete "
            "object, count, location, cause, or impact that the source description does not "
            "support. Return only the final caption and no reasoning."
        )
    else:
        instruction = (
            "Refine and enrich the description while preserving its original meaning. Improve "
            "clarity and specificity, but do not invent objects, counts, locations, causes, or "
            "effects that are not supported by the original description. Return only one final "
            "caption with no explanation."
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a professional remote sensing change detection caption editor. "
                + instruction
            ),
        },
        {
            "role": "user",
            "content": "Original change description:\n{}".format(original_caption.strip()),
        },
    ]


def image_text_caption_messages(
    original_caption: str, images: EncodedImagePair
) -> List[ChatMessage]:
    if not original_caption.strip():
        raise ValueError("original caption cannot be empty")
    content: List[Dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Image A is before and image B is after. Compare the co-registered remote-sensing "
                "images and expand the reference description into one accurate change caption. "
                "The reference is a fallible model output: preserve supported information, correct "
                "claims that conflict with the images, and add only visually supported objects, "
                "counts, directions, or locations. Ignore illumination, color balance, shadows, "
                "and seasonal appearance unless they accompany structural change. Return only the "
                "final caption.\n\nReference description:\n{}"
            ).format(original_caption.strip()),
        },
        {"type": "image_url", "image_url": {"url": images.before_data_url}},
        {"type": "image_url", "image_url": {"url": images.after_data_url}},
    ]
    return [
        {
            "role": "system",
            "content": "You write concise, image-grounded bi-temporal remote-sensing captions.",
        },
        {"role": "user", "content": content},
    ]


def clean_caption_output(content: str) -> str:
    value = content.strip()
    match = re.search(r"\{[\s\S]*\}", value)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("refined_caption", "caption", "final_caption"):
                candidate = payload.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    value = candidate.strip()
                    break
    value = re.sub(r"^```(?:text|markdown)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(
        r"^(?:new caption|final caption|refined caption|caption)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip().strip('"').strip()
