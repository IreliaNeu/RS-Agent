"""Versioned prompts for the paper RS-CC caption enrichment task."""

from __future__ import annotations

import json
import re
from typing import List

from rs_agent.domains.remote_sensing.config import CaptionPromptProfile
from rs_agent.providers.base import ChatMessage

BASE_PROMPT_VERSION = "rs_cc_enrichment_base_v1"
COT_PROMPT_VERSION = "rs_cc_enrichment_cot_no_background_v1"


def prompt_version(profile: CaptionPromptProfile) -> str:
    if profile == CaptionPromptProfile.COT_WITHOUT_BACKGROUND:
        return COT_PROMPT_VERSION
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
        r"^(?:final caption|refined caption|caption)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip().strip('"').strip()
