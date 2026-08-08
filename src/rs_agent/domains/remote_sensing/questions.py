"""Preset and user-question policy for remote-sensing change VQA."""

from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from rs_agent.core.schemas import QuestionSource, VQAQuestion

QUESTION_TEMPLATES: Dict[str, str] = {
    "change_summary": (
        "What meaningful land-surface changes occurred between image A (before) "
        "and image B (after)? Distinguish structural change from seasonal or color differences."
    ),
    "change_presence": (
        "Is there a meaningful land-surface change between image A and image B? "
        "Answer yes or no, then cite visible evidence."
    ),
    "structural_change": (
        "Did any building, road, water boundary, or land-cover layout appear, disappear, "
        "or change shape? Do not count illumination or seasonal color alone."
    ),
    "building_change": (
        "Were any buildings newly constructed, removed, or visibly altered between "
        "image A and image B?"
    ),
    "road_change": (
        "Were any roads newly built, removed, widened, or otherwise changed between "
        "image A and image B?"
    ),
    "vegetation_change": (
        "Did the spatial extent or layout of vegetation, forest, or cultivated land change, "
        "beyond seasonal color or phenology?"
    ),
    "water_change": (
        "Did the visible extent or course of any water body change between image A and image B?"
    ),
    "spatial_location": (
        "Where are the main changes located in the scene, and how spatially extensive are they?"
    ),
}


_MATCH_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("building_change", ("building", "house", "urban", "construct", "建筑", "房屋", "城市")),
    ("road_change", ("road", "street", "highway", "道路", "公路")),
    ("vegetation_change", ("vegetation", "forest", "farmland", "crop", "植被", "森林", "农田")),
    ("water_change", ("water", "river", "lake", "pond", "水体", "河流", "湖")),
    ("spatial_location", ("where", "location", "extent", "position", "哪里", "位置", "范围")),
    ("structural_change", ("structural", "structure", "layout", "结构", "布局")),
    ("change_presence", ("any change", "whether", "有无变化", "是否变化", "发生变化")),
)


def match_template(question: str) -> Optional[str]:
    normalized = question.casefold()
    for template_id, keywords in _MATCH_RULES:
        if any(keyword.casefold() in normalized for keyword in keywords):
            return template_id
    return None


def resolve_questions(
    user_questions: Iterable[str], default_template_ids: Iterable[str]
) -> List[VQAQuestion]:
    cleaned: List[str] = []
    seen = set()
    for question in user_questions:
        text = question.strip()
        key = text.casefold()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)

    if cleaned:
        return [
            VQAQuestion(
                text=text,
                source=QuestionSource.HYBRID if match_template(text) else QuestionSource.USER,
                template_id=match_template(text),
            )
            for text in cleaned
        ]

    questions: List[VQAQuestion] = []
    for template_id in default_template_ids:
        if template_id not in QUESTION_TEMPLATES:
            raise ValueError("unknown RS-VQA question template: {}".format(template_id))
        questions.append(
            VQAQuestion(
                text=QUESTION_TEMPLATES[template_id],
                source=QuestionSource.TEMPLATE,
                template_id=template_id,
            )
        )
    if not questions:
        raise ValueError("RS-VQA requires user questions or default templates")
    return questions
