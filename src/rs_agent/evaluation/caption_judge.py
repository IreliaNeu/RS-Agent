"""LLM-as-Judge evaluation preserving the paper experiment semantics."""

from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from rs_agent.core.config import ModelConfig
from rs_agent.core.schemas import (
    CaptionCandidate,
    CaptionSelection,
    ModelResponse,
    StrictModel,
)
from rs_agent.evaluation.parsing import parse_choice, parse_scores, select_caption
from rs_agent.evaluation.prompts import best_choice_messages, score_messages
from rs_agent.providers.base import ChatProvider


class CaptionEvaluationRecord(StrictModel):
    original_caption: str
    candidates: List[CaptionCandidate]
    judge_choice: str = ""
    scores: Dict[str, int] = Field(default_factory=dict)
    selection: CaptionSelection
    selector_response: ModelResponse
    evaluator_response: ModelResponse

    def legacy_full_record(self) -> Dict[str, object]:
        return {
            "Original Caption": self.original_caption,
            "Options": {candidate.label: candidate.text for candidate in self.candidates},
            "Judge Choice": self.judge_choice or None,
            "Scores": {
                score.label: score.score for score in self.selection.scores
            },
            "Selected Label": self.selection.selected_label,
            "Selection Policy": self.selection.policy,
        }

    def legacy_best_record(self) -> Dict[str, object]:
        selected = next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selection.selected_candidate_id
        )
        return {
            "Original Caption": self.original_caption,
            "Best Caption": selected.text,
            "Selected Label": self.selection.selected_label,
        }


class CaptionJudge:
    def __init__(
        self,
        selector_provider: ChatProvider,
        selector_model: ModelConfig,
        evaluator_provider: ChatProvider,
        evaluator_model: ModelConfig,
    ):
        self.selector_provider = selector_provider
        self.selector_model = selector_model
        self.evaluator_provider = evaluator_provider
        self.evaluator_model = evaluator_model

    async def evaluate(
        self, original_caption: str, candidates: List[CaptionCandidate]
    ) -> CaptionEvaluationRecord:
        if not original_caption.strip():
            raise ValueError("original caption is required for caption evaluation")
        if not candidates:
            raise ValueError("caption evaluation requires candidates")
        options = {candidate.label.upper(): candidate.text for candidate in candidates}
        if len(options) != len(candidates):
            raise ValueError("candidate labels must be unique")

        selector_response = await self.selector_provider.complete(
            messages=best_choice_messages(original_caption, options),
            model=self.selector_model.model,
            temperature=self.selector_model.temperature,
            max_tokens=self.selector_model.max_tokens,
            metadata=self.selector_model.request_options or None,
        )
        judge_choice = parse_choice(selector_response.content, options)

        evaluator_response = await self.evaluator_provider.complete(
            messages=score_messages(original_caption, options),
            model=self.evaluator_model.model,
            temperature=self.evaluator_model.temperature,
            max_tokens=self.evaluator_model.max_tokens,
            metadata=self.evaluator_model.request_options or None,
        )
        parsed_scores = parse_scores(evaluator_response.content, options)
        selection = select_caption(candidates, parsed_scores, judge_choice)
        integer_scores = {
            label: score for label, score in parsed_scores.items() if score is not None
        }

        return CaptionEvaluationRecord(
            original_caption=original_caption,
            candidates=candidates,
            judge_choice=judge_choice or "",
            scores=integer_scores,
            selection=selection,
            selector_response=selector_response,
            evaluator_response=evaluator_response,
        )
