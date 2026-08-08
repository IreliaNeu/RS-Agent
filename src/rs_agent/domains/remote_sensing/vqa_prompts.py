"""Multimodal prompts for RS-VQA generation and evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

from rs_agent.domains.remote_sensing.image_inputs import EncodedImagePair
from rs_agent.providers.base import ChatMessage

CHANGE_DEFINITION = (
    "Meaningful structural change means an addition, removal, or spatial reconfiguration of "
    "objects or land-cover boundaries, such as buildings, roads, water, or cleared land. "
    "Illumination, sensor color balance, shadows, and seasonal phenology alone are not "
    "structural change."
)


def _image_parts(images: EncodedImagePair) -> List[Dict[str, object]]:
    return [
        {"type": "image_url", "image_url": {"url": images.before_data_url}},
        {"type": "image_url", "image_url": {"url": images.after_data_url}},
    ]


def answer_messages(
    question: str,
    images: EncodedImagePair,
    knowledge_caption: Optional[str] = None,
) -> List[ChatMessage]:
    context = ""
    if knowledge_caption and knowledge_caption.strip():
        context = (
            "\nAuxiliary caption C* from the text-only RS-CC stage:\n{}\n"
            "Treat C* as a hypothesis and correct it whenever it conflicts with the images."
        ).format(knowledge_caption.strip())
    content: List[Dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Image A is the before image and image B is the after image. "
                "Answer using visible evidence from the original pair. {}{}\nQuestion:\n{}"
            ).format(CHANGE_DEFINITION, context, question),
        }
    ]
    content.extend(_image_parts(images))
    return [
        {
            "role": "system",
            "content": (
                "You are a remote-sensing visual question answering expert. Compare the two "
                "co-registered images carefully. Be concise, separate observation from "
                "uncertainty, and do not infer causes or objects without visible support."
            ),
        },
        {"role": "user", "content": content},
    ]


def _options_text(options: Dict[str, str]) -> str:
    return "\n".join("{}. {}".format(label, answer) for label, answer in options.items())


def best_answer_messages(
    question: str, options: Dict[str, str], images: EncodedImagePair
) -> List[ChatMessage]:
    content: List[Dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Question:\n{}\n\nCandidate answers:\n{}\n\n{} Select the answer that is most "
                "accurate, complete, and grounded in the visible image pair. Penalize claims "
                "that convert color or seasonal differences into unsupported construction or "
                "land-cover change. Return only one candidate letter."
            ).format(question, _options_text(options), CHANGE_DEFINITION),
        }
    ]
    content.extend(_image_parts(images))
    return [
        {
            "role": "system",
            "content": "You are an expert evaluator of remote-sensing change VQA answers.",
        },
        {"role": "user", "content": content},
    ]


def score_answer_messages(
    question: str, options: Dict[str, str], images: EncodedImagePair
) -> List[ChatMessage]:
    content: List[Dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Question:\n{}\n\nCandidate answers:\n{}\n\n{} Score every answer from 1 "
                "(worst) to 10 (best) for visual correctness, relevance, completeness, and "
                "absence of unsupported claims. Strongly penalize invented objects and treating "
                "seasonal appearance alone as structural change. Return only a JSON object such "
                "as {{\"A\": 8, \"B\": 7}}."
            ).format(question, _options_text(options), CHANGE_DEFINITION),
        }
    ]
    content.extend(_image_parts(images))
    return [
        {
            "role": "system",
            "content": "You score remote-sensing VQA answers against the original image pair.",
        },
        {"role": "user", "content": content},
    ]
