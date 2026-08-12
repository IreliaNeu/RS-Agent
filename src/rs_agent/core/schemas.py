"""Typed contracts shared by the RS-Agent pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskType(str, Enum):
    CAPTION_ENRICHMENT = "caption_enrichment"
    CAPTION_FROM_SCRATCH = "caption_from_scratch"
    VQA = "vqa"
    COMBINED = "combined"


class QuestionSource(str, Enum):
    USER = "user"
    TEMPLATE = "template"
    HYBRID = "hybrid"


class ImagePair(StrictModel):
    before: Path
    after: Path


class TaskRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    task_type: TaskType
    images: ImagePair
    original_caption: Optional[str] = None
    user_questions: List[str] = Field(default_factory=list)
    use_knowledge_bridge: bool = True
    request_mask: bool = False

    @model_validator(mode="after")
    def validate_task_inputs(self) -> "TaskRequest":
        if self.task_type in {TaskType.CAPTION_ENRICHMENT, TaskType.COMBINED}:
            if not self.original_caption or not self.original_caption.strip():
                raise ValueError("caption enrichment requires one original pair caption")
        return self


class ModelRef(StrictModel):
    name: str
    provider: str
    model: str


class TokenUsage(StrictModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class RequestTelemetry(StrictModel):
    attempts: int = Field(default=1, ge=1)
    latency_ms: float = Field(default=0.0, ge=0.0)
    status_code: Optional[int] = None
    attempt_status_codes: List[int] = Field(default_factory=list)


class ModelResponse(StrictModel):
    provider: str
    model: str
    content: str
    response_id: Optional[str] = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    telemetry: RequestTelemetry = Field(default_factory=RequestTelemetry)
    assistant_message: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CaptionCandidate(StrictModel):
    candidate_id: str = Field(default_factory=lambda: uuid4().hex)
    label: str
    model: ModelRef
    input_mode: str = "text_only"
    text: str
    response: Optional[ModelResponse] = None
    telemetry: Optional[RequestTelemetry] = None
    error: Optional[str] = None


class CandidateScore(StrictModel):
    label: str
    score: Optional[int] = Field(default=None, ge=1, le=10)


class CaptionSelection(StrictModel):
    policy: str = "highest_score_then_judge"
    selected_label: str
    selected_candidate_id: str
    judge_choice: Optional[str] = None
    scores: List[CandidateScore]


class VQAQuestion(StrictModel):
    question_id: str = Field(default_factory=lambda: uuid4().hex)
    text: str
    source: QuestionSource
    template_id: Optional[str] = None


class VQAAnswer(StrictModel):
    question_id: str
    text: str
    model: ModelRef
    response: Optional[ModelResponse] = None
    error: Optional[str] = None


class VQAAnswerCandidate(StrictModel):
    candidate_id: str = Field(default_factory=lambda: uuid4().hex)
    question_id: str
    label: str
    text: str
    model: ModelRef
    response: Optional[ModelResponse] = None
    telemetry: Optional[RequestTelemetry] = None
    error: Optional[str] = None


class VQAAnswerSelection(StrictModel):
    policy: str = "highest_score_then_judge"
    question_id: str
    selected_label: str
    selected_candidate_id: str
    judge_choice: Optional[str] = None
    scores: List[CandidateScore]


class PipelineResult(StrictModel):
    request_id: str
    run_id: str
    caption_candidates: List[CaptionCandidate] = Field(default_factory=list)
    caption_selection: Optional[CaptionSelection] = None
    questions: List[VQAQuestion] = Field(default_factory=list)
    answers: List[VQAAnswer] = Field(default_factory=list)
    mask_artifact_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
