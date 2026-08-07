"""Prompts migrated from the paper's LLM-as-Judge evaluation script."""

from __future__ import annotations

from typing import Dict, List

from rs_agent.providers.base import ChatMessage


def _choices_text(options: Dict[str, str]) -> str:
    return "\n".join("{}. {}".format(label, text) for label, text in options.items())


def best_choice_messages(original_caption: str, options: Dict[str, str]) -> List[ChatMessage]:
    return [
        {
            "role": "system",
            "content": "You are an expert evaluator of remote sensing change descriptions.",
        },
        {
            "role": "user",
            "content": (
                "Original change description:\n{}\n\nCandidate enriched descriptions:\n{}\n\n"
                "Choose the candidate that provides the most useful information while remaining "
                "faithful to the original description. Return only one candidate letter."
            ).format(original_caption, _choices_text(options)),
        },
    ]


def score_messages(original_caption: str, options: Dict[str, str]) -> List[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator. Score each remote sensing change caption from 1 "
                "(worst) to 10 (best) for information richness and fidelity to the original."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original change description:\n{}\n\nCandidates:\n{}\n\n"
                "Return a JSON object mapping every candidate letter to one integer score, for "
                "example {{\"A\": 8, \"B\": 7}}. Return no additional text."
            ).format(original_caption, _choices_text(options)),
        },
    ]

