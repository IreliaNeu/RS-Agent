"""Domain-neutral evidence bundle and conservative conflict auditing."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from rs_agent.core.schemas import StrictModel
from rs_agent.domains.remote_sensing.mask_evidence import MaskEvidenceSummary
from rs_agent.orchestration.caption_pipeline import RSCCPipelineResult
from rs_agent.orchestration.vqa_pipeline import RSVQAPipelineResult


class EvidenceStance(str, Enum):
    CHANGE = "change"
    NO_CHANGE = "no_change"
    UNCERTAIN = "uncertain"


class EvidenceKind(str, Enum):
    ORIGINAL_CAPTION = "original_caption"
    SELECTED_CAPTION = "selected_caption"
    VQA_ANSWER = "vqa_answer"
    MASK = "mask"


class EvidenceClaim(StrictModel):
    source_id: str
    kind: EvidenceKind
    statement: str
    stance: EvidenceStance
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceConflict(StrictModel):
    left_source_id: str
    right_source_id: str
    reason: str


class EvidenceBundle(StrictModel):
    claims: List[EvidenceClaim]
    conflicts: List[EvidenceConflict]
    consensus: EvidenceStance
    has_conflict: bool


_NO_CHANGE_PATTERNS = (
    r"\bno\s+(?:(?:significant|discernible|meaningful|notable|visible|apparent|major)\s+)?(?:(?:structural|land[- ]surface)\s+)?changes?\b",
    r"\bwithout\s+(?:any\s+)?(?:significant\s+)?changes?\b",
    r"\bunchanged\b",
    r"\bsame\s+as\s+before\b",
    r"\bno\s+difference\b",
    r"没有(?:明显)?变化",
    r"未发生变化",
    r"保持不变",
)
_CHANGE_PATTERNS = (
    r"\bchanges?\b",
    r"\bchanged\b",
    r"\bnew\b",
    r"\bappear(?:s|ed)?\b",
    r"\bbuilt\b",
    r"\bconstruct(?:ed|ion)?\b",
    r"\bremove(?:d)?\b",
    r"\bdemolish(?:ed)?\b",
    r"\breplac(?:e|ed)\b",
    r"\bexpand(?:ed|sion)?\b",
    r"新增",
    r"新建",
    r"出现",
    r"移除",
    r"拆除",
    r"替换",
    r"扩建",
    r"发生变化",
)


def classify_statement(text: str) -> EvidenceStance:
    normalized = " ".join(text.strip().lower().split())
    if any(re.search(pattern, normalized) for pattern in _NO_CHANGE_PATTERNS):
        return EvidenceStance.NO_CHANGE
    if any(re.search(pattern, normalized) for pattern in _CHANGE_PATTERNS):
        return EvidenceStance.CHANGE
    return EvidenceStance.UNCERTAIN


def build_evidence_bundle(
    *,
    original_caption: str,
    caption: Optional[RSCCPipelineResult],
    vqa: Optional[RSVQAPipelineResult],
    mask: Optional[MaskEvidenceSummary],
) -> EvidenceBundle:
    claims = [
        EvidenceClaim(
            source_id="caption:original",
            kind=EvidenceKind.ORIGINAL_CAPTION,
            statement=original_caption,
            stance=classify_statement(original_caption),
        )
    ]
    if caption is not None:
        claims.append(
            EvidenceClaim(
                source_id="caption:c_star",
                kind=EvidenceKind.SELECTED_CAPTION,
                statement=caption.selected_caption,
                stance=classify_statement(caption.selected_caption),
                metadata={"selected_label": caption.selected_label},
            )
        )
    if vqa is not None:
        for selected in vqa.selected_answers:
            claims.append(
                EvidenceClaim(
                    source_id="vqa:{}".format(selected.question.question_id),
                    kind=EvidenceKind.VQA_ANSWER,
                    statement=selected.answer,
                    stance=classify_statement(selected.answer),
                    metadata={
                        "question": selected.question.text,
                        "model_name": selected.model_name,
                    },
                )
            )
    if mask is not None:
        claims.append(
            EvidenceClaim(
                source_id="mask:{}".format(mask.sha256[:12]),
                kind=EvidenceKind.MASK,
                statement=mask.interpretation,
                stance=(
                    EvidenceStance.CHANGE
                    if mask.has_change
                    else EvidenceStance.NO_CHANGE
                ),
                metadata={
                    "source": mask.source.value,
                    "changed_ratio": mask.changed_ratio,
                    "significant_component_count": mask.significant_component_count,
                },
            )
        )

    change_claims = [claim for claim in claims if claim.stance == EvidenceStance.CHANGE]
    no_change_claims = [
        claim for claim in claims if claim.stance == EvidenceStance.NO_CHANGE
    ]
    conflicts = [
        EvidenceConflict(
            left_source_id=changed.source_id,
            right_source_id=unchanged.source_id,
            reason="One source indicates change while the other indicates no change.",
        )
        for changed in change_claims
        for unchanged in no_change_claims
    ]
    if conflicts or (not change_claims and not no_change_claims):
        consensus = EvidenceStance.UNCERTAIN
    elif change_claims:
        consensus = EvidenceStance.CHANGE
    else:
        consensus = EvidenceStance.NO_CHANGE
    return EvidenceBundle(
        claims=claims,
        conflicts=conflicts,
        consensus=consensus,
        has_conflict=bool(conflicts),
    )
