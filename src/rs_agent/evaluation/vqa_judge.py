"""Image-grounded LLM-as-Judge evaluation for RS-VQA candidates."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from rs_agent.core.config import ModelConfig
from rs_agent.core.schemas import (
    CandidateScore,
    ModelResponse,
    StrictModel,
    VQAAnswerCandidate,
    VQAAnswerSelection,
    VQAQuestion,
)
from rs_agent.domains.remote_sensing.image_inputs import EncodedImagePair
from rs_agent.domains.remote_sensing.vqa_prompts import (
    best_answer_messages,
    score_answer_messages,
)
from rs_agent.evaluation.parsing import parse_choice, parse_scores
from rs_agent.providers.base import ChatProvider


class VQAEvaluationRecord(StrictModel):
    question: VQAQuestion
    candidates: List[VQAAnswerCandidate]
    judge_choice: str = ""
    scores: Dict[str, int] = Field(default_factory=dict)
    selection: VQAAnswerSelection
    selector_response: ModelResponse
    evaluator_response: ModelResponse


def select_vqa_answer(
    question: VQAQuestion,
    candidates: List[VQAAnswerCandidate],
    scores: Dict[str, Optional[int]],
    judge_choice: Optional[str],
) -> VQAAnswerSelection:
    if not candidates:
        raise ValueError("at least one VQA candidate is required")
    by_label = {candidate.label.upper(): candidate for candidate in candidates}
    if len(by_label) != len(candidates):
        raise ValueError("VQA candidate labels must be unique")
    valid_scores = {
        label: score
        for label, score in scores.items()
        if label in by_label and score is not None
    }
    choice = judge_choice.upper() if judge_choice else None
    if valid_scores:
        highest = max(valid_scores.values())
        tied = [
            candidate.label.upper()
            for candidate in candidates
            if valid_scores.get(candidate.label.upper()) == highest
        ]
        selected_label = choice if choice in tied else tied[0]
    else:
        selected_label = choice if choice in by_label else candidates[0].label.upper()
    selected = by_label[selected_label]
    return VQAAnswerSelection(
        question_id=question.question_id,
        selected_label=selected_label,
        selected_candidate_id=selected.candidate_id,
        judge_choice=judge_choice,
        scores=[
            CandidateScore(
                label=candidate.label.upper(),
                score=scores.get(candidate.label.upper()),
            )
            for candidate in candidates
        ],
    )


class VQAJudge:
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
        self,
        question: VQAQuestion,
        candidates: List[VQAAnswerCandidate],
        images: EncodedImagePair,
    ) -> VQAEvaluationRecord:
        options = {candidate.label.upper(): candidate.text for candidate in candidates}
        if len(options) != len(candidates):
            raise ValueError("VQA candidate labels must be unique")
        selector_response = await self.selector_provider.complete(
            messages=best_answer_messages(question.text, options, images),
            model=self.selector_model.model,
            temperature=self.selector_model.temperature,
            max_tokens=self.selector_model.max_tokens,
            metadata=self.selector_model.request_options or None,
        )
        judge_choice = parse_choice(selector_response.content, options)
        evaluator_response = await self.evaluator_provider.complete(
            messages=score_answer_messages(question.text, options, images),
            model=self.evaluator_model.model,
            temperature=self.evaluator_model.temperature,
            max_tokens=self.evaluator_model.max_tokens,
            metadata=self.evaluator_model.request_options or None,
        )
        parsed_scores = parse_scores(evaluator_response.content, options)
        selection = select_vqa_answer(question, candidates, parsed_scores, judge_choice)
        return VQAEvaluationRecord(
            question=question,
            candidates=candidates,
            judge_choice=judge_choice or "",
            scores={
                label: score
                for label, score in parsed_scores.items()
                if score is not None
            },
            selection=selection,
            selector_response=selector_response,
            evaluator_response=evaluator_response,
        )
