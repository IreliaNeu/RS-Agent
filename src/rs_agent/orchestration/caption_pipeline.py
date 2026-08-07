"""Paper-aligned RS-CC generation, judging, selection, and artifact workflow."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.core.schemas import CaptionCandidate, StrictModel
from rs_agent.domains.remote_sensing.caption_agent import (
    CaptionGenerationBatch,
    RSCCAgent,
    RSCCRequest,
)
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.evaluation.caption_judge import CaptionEvaluationRecord, CaptionJudge
from rs_agent.providers.registry import ProviderRegistry


class InsufficientCandidatesError(RuntimeError):
    def __init__(self, message: str, generation_artifact: Path):
        super().__init__(message)
        self.generation_artifact = generation_artifact


class RSCCPipelineResult(StrictModel):
    run_id: str
    request_id: str
    item_id: str
    selected_label: str
    selected_caption: str
    generation_artifact: str
    evaluation_artifact: str
    result_artifact: str
    candidate_count: int = Field(ge=1)


class RSCCPipeline:
    def __init__(
        self,
        config: RSCCExperimentConfig,
        providers: ProviderRegistry,
        artifacts: JsonArtifactStore,
    ):
        self.config = config
        self.providers = providers
        self.artifacts = artifacts
        self.agent = RSCCAgent(config=config, providers=providers)
        self.judge = CaptionJudge(
            selector_provider=providers.get(config.selector.provider),
            selector_model=config.selector,
            evaluator_provider=providers.get(config.evaluator.provider),
            evaluator_model=config.evaluator,
        )

    async def run(self, request: RSCCRequest, run_id: str) -> RSCCPipelineResult:
        generation = await self.agent.generate(request)
        generation_artifact = self._write_generation(generation, run_id)
        successful = generation.successful_candidates
        if len(successful) < self.config.minimum_successful_candidates:
            raise InsufficientCandidatesError(
                "RS-CC produced {} successful candidates; profile requires {}".format(
                    len(successful), self.config.minimum_successful_candidates
                ),
                generation_artifact=generation_artifact,
            )

        evaluation = await self.judge.evaluate(request.original_caption, successful)
        evaluation_artifact = self._write_evaluation(evaluation, request, run_id)
        selected = self._selected_candidate(evaluation, successful)
        result_payload = {
            "request_id": request.request_id,
            "item_id": request.item_id,
            "selected_label": evaluation.selection.selected_label,
            "selected_caption": selected.text,
            "selection": evaluation.selection.model_dump(mode="json"),
            "source_artifacts": {
                "generation": str(generation_artifact),
                "evaluation": str(evaluation_artifact),
            },
        }
        result_envelope = ArtifactEnvelope.create(
            artifact_type="rs_cc_result",
            run_id=run_id,
            item_id=request.item_id,
            payload=result_payload,
            metadata={"profile": self.config.profile},
        )
        result_artifact = self.artifacts.write(result_envelope)
        return RSCCPipelineResult(
            run_id=run_id,
            request_id=request.request_id,
            item_id=request.item_id,
            selected_label=evaluation.selection.selected_label,
            selected_caption=selected.text,
            generation_artifact=str(generation_artifact),
            evaluation_artifact=str(evaluation_artifact),
            result_artifact=str(result_artifact),
            candidate_count=len(successful),
        )

    def _write_generation(self, batch: CaptionGenerationBatch, run_id: str) -> Path:
        envelope = ArtifactEnvelope.create(
            artifact_type="caption_candidates",
            run_id=run_id,
            item_id=batch.item_id,
            payload=batch.model_dump(mode="json"),
            metadata={
                "profile": self.config.profile,
                "prompt_version": batch.prompt_version,
            },
        )
        return self.artifacts.write(envelope)

    def _write_evaluation(
        self, evaluation: CaptionEvaluationRecord, request: RSCCRequest, run_id: str
    ) -> Path:
        envelope = ArtifactEnvelope.create(
            artifact_type="caption_evaluation",
            run_id=run_id,
            item_id=request.item_id,
            payload=evaluation.model_dump(mode="json"),
            metadata={
                "selector": self.config.selector.model,
                "evaluator": self.config.evaluator.model,
            },
        )
        return self.artifacts.write(envelope)

    @staticmethod
    def _selected_candidate(
        evaluation: CaptionEvaluationRecord, candidates: List[CaptionCandidate]
    ) -> CaptionCandidate:
        return next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == evaluation.selection.selected_candidate_id
        )
