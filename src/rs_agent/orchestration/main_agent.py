"""Paper-aligned Main Agent task planning.

The paper defines the Main Agent as a coordinator.  This module therefore
produces an auditable execution plan; it does not add an unreported LLM
fusion step to the experimental baseline.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import Field

from rs_agent.core.schemas import StrictModel, TaskType


class PipelineStage(str, Enum):
    RS_CC = "rs_cc"
    KNOWLEDGE_BRIDGE = "knowledge_bridge"
    RS_VQA = "rs_vqa"
    MASK_EVIDENCE = "mask_evidence"


class MainAgentPlan(StrictModel):
    """The stages selected by the Main Agent for one request."""

    task_type: TaskType
    stages: List[PipelineStage] = Field(min_length=1)
    return_caption: bool
    return_answers: bool
    use_knowledge_bridge: bool
    include_mask: bool
    rationale: str


def plan_request(
    task_type: TaskType,
    *,
    use_knowledge_bridge: bool,
    has_mask: bool,
    has_provided_knowledge: bool = False,
) -> MainAgentPlan:
    """Translate the requested output into the smallest valid paper workflow."""

    if task_type == TaskType.CAPTION_FROM_SCRATCH:
        raise ValueError(
            "caption_from_scratch is outside the current LEVIR-MCI profile; "
            "provide the required original pair caption and use caption_enrichment"
        )

    stages: List[PipelineStage] = []
    return_caption = task_type in {
        TaskType.CAPTION_ENRICHMENT,
        TaskType.COMBINED,
    }
    return_answers = task_type in {TaskType.VQA, TaskType.COMBINED}

    # VQA needs RS-CC only when C* is requested through the bridge.  This
    # preserves the paper's with/without-KB ablation as a real execution path.
    run_caption = return_caption or (
        return_answers and use_knowledge_bridge and not has_provided_knowledge
    )
    if run_caption:
        stages.append(PipelineStage.RS_CC)
    if return_answers and use_knowledge_bridge:
        stages.append(PipelineStage.KNOWLEDGE_BRIDGE)
    if return_answers:
        stages.append(PipelineStage.RS_VQA)
    if has_mask:
        stages.append(PipelineStage.MASK_EVIDENCE)

    if return_answers and use_knowledge_bridge and has_provided_knowledge:
        rationale = "RS-VQA requested with a previously selected, provenance-linked C*."
    elif return_answers and use_knowledge_bridge:
        rationale = "RS-VQA requested with selected caption C* as contextual prior."
    elif return_answers:
        rationale = "RS-VQA requested without caption context for the KB ablation path."
    else:
        rationale = "Informative change caption requested; RS-VQA is not required."
    if has_mask:
        rationale += " An external change mask is retained as optional evidence."

    return MainAgentPlan(
        task_type=task_type,
        stages=stages,
        return_caption=return_caption,
        return_answers=return_answers,
        use_knowledge_bridge=return_answers and use_knowledge_bridge,
        include_mask=has_mask,
        rationale=rationale,
    )
