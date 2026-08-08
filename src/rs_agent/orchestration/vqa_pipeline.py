"""Question resolution, five-model RS-VQA, evaluation, and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import Field

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.core.schemas import StrictModel, VQAAnswerCandidate, VQAAnswerSelection, VQAQuestion
from rs_agent.domains.remote_sensing.image_inputs import encode_original_image_pair
from rs_agent.domains.remote_sensing.questions import resolve_questions
from rs_agent.domains.remote_sensing.vqa_agent import RSVQAAgent, RSVQARequest, VQAGenerationBatch
from rs_agent.domains.remote_sensing.vqa_config import RSVQAExperimentConfig
from rs_agent.evaluation.vqa_judge import VQAEvaluationRecord, VQAJudge
from rs_agent.providers.registry import ProviderRegistry


class InsufficientVQAAnswersError(RuntimeError):
    def __init__(self, message: str, generation_artifact: Path):
        super().__init__(message)
        self.generation_artifact = generation_artifact


class SelectedVQAAnswer(StrictModel):
    question: VQAQuestion
    answer: str
    model_name: str
    selection: VQAAnswerSelection


class RSVQAPipelineResult(StrictModel):
    run_id: str
    request_id: str
    item_id: str
    selected_answers: List[SelectedVQAAnswer]
    generation_artifact: str
    evaluation_artifact: str
    result_artifact: str
    question_count: int = Field(ge=1)


class RSVQAPipeline:
    def __init__(
        self,
        config: RSVQAExperimentConfig,
        providers: ProviderRegistry,
        artifacts: JsonArtifactStore,
    ):
        self.config = config
        self.providers = providers
        self.artifacts = artifacts
        self.agent = RSVQAAgent(config, providers)
        self.judge = VQAJudge(
            selector_provider=providers.get(config.selector.provider),
            selector_model=config.selector,
            evaluator_provider=providers.get(config.evaluator.provider),
            evaluator_model=config.evaluator,
        )

    async def run(self, request: RSVQARequest, run_id: str) -> RSVQAPipelineResult:
        questions = resolve_questions(
            request.user_questions, self.config.default_template_ids
        )
        images = encode_original_image_pair(request.images)
        generation = await self.agent.generate(request, questions, images)
        generation_artifact = self._write_generation(generation, run_id)

        by_question: Dict[str, List[VQAAnswerCandidate]] = {
            question.question_id: [] for question in questions
        }
        for candidate in generation.candidates:
            if not candidate.error:
                by_question[candidate.question_id].append(candidate)
        failures = {
            question.question_id: len(by_question[question.question_id])
            for question in questions
            if len(by_question[question.question_id])
            < self.config.minimum_successful_answers_per_question
        }
        if failures:
            raise InsufficientVQAAnswersError(
                "RS-VQA has insufficient candidates for questions: {}".format(failures),
                generation_artifact,
            )

        evaluations = []
        for question in questions:
            evaluations.append(
                await self.judge.evaluate(
                    question,
                    by_question[question.question_id],
                    images,
                )
            )
        evaluation_artifact = self._write_evaluations(
            request, evaluations, images.model_dump(), run_id
        )
        selected_answers = [self._selected_answer(record) for record in evaluations]
        result_envelope = ArtifactEnvelope.create(
            artifact_type="rs_vqa_result",
            run_id=run_id,
            item_id=request.item_id,
            payload={
                "request_id": request.request_id,
                "item_id": request.item_id,
                "selected_answers": [
                    answer.model_dump(mode="json") for answer in selected_answers
                ],
                "source_artifacts": {
                    "generation": str(generation_artifact),
                    "evaluation": str(evaluation_artifact),
                },
            },
            metadata={"profile": self.config.profile},
        )
        result_artifact = self.artifacts.write(result_envelope)
        return RSVQAPipelineResult(
            run_id=run_id,
            request_id=request.request_id,
            item_id=request.item_id,
            selected_answers=selected_answers,
            generation_artifact=str(generation_artifact),
            evaluation_artifact=str(evaluation_artifact),
            result_artifact=str(result_artifact),
            question_count=len(questions),
        )

    def _write_generation(self, batch: VQAGenerationBatch, run_id: str) -> Path:
        envelope = ArtifactEnvelope.create(
            artifact_type="vqa_candidates",
            run_id=run_id,
            item_id=batch.item_id,
            payload=batch.model_dump(mode="json"),
            metadata={"profile": self.config.profile},
        )
        return self.artifacts.write(envelope)

    def _write_evaluations(
        self,
        request: RSVQARequest,
        records: List[VQAEvaluationRecord],
        image_metadata: Dict[str, str],
        run_id: str,
    ) -> Path:
        envelope = ArtifactEnvelope.create(
            artifact_type="vqa_evaluation",
            run_id=run_id,
            item_id=request.item_id,
            payload={
                "request_id": request.request_id,
                "item_id": request.item_id,
                "records": [record.model_dump(mode="json") for record in records],
                "image_sha256": {
                    "before": image_metadata["before_sha256"],
                    "after": image_metadata["after_sha256"],
                },
            },
            metadata={
                "selector": self.config.selector.model,
                "evaluator": self.config.evaluator.model,
            },
        )
        return self.artifacts.write(envelope)

    @staticmethod
    def _selected_answer(record: VQAEvaluationRecord) -> SelectedVQAAnswer:
        candidate = next(
            item
            for item in record.candidates
            if item.candidate_id == record.selection.selected_candidate_id
        )
        return SelectedVQAAnswer(
            question=record.question,
            answer=candidate.text,
            model_name=candidate.model.name,
            selection=record.selection,
        )
