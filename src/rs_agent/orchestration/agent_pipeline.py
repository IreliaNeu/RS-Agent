"""Paper-aligned Main Agent orchestration for RS-CC, RS-VQA, and masks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.core.schemas import ImagePair, StrictModel, TaskType
from rs_agent.domains.remote_sensing.caption_agent import RSCCRequest
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.domains.remote_sensing.mask_evidence import (
    MaskEvidenceRequest,
    MaskEvidenceSummary,
    analyze_mask,
)
from rs_agent.domains.remote_sensing.vqa_agent import RSVQARequest
from rs_agent.domains.remote_sensing.vqa_config import RSVQAExperimentConfig
from rs_agent.orchestration.caption_pipeline import RSCCPipeline, RSCCPipelineResult
from rs_agent.orchestration.evidence import EvidenceBundle, build_evidence_bundle
from rs_agent.orchestration.knowledge_bridge import (
    KnowledgeBridgePacket,
    bridge_selected_caption,
)
from rs_agent.orchestration.main_agent import MainAgentPlan, PipelineStage, plan_request
from rs_agent.orchestration.vqa_pipeline import RSVQAPipeline, RSVQAPipelineResult
from rs_agent.providers.registry import ProviderRegistry


class RSAgentRequest(StrictModel):
    item_id: str
    original_caption: str
    images: ImagePair
    task_type: TaskType = TaskType.COMBINED
    user_questions: List[str] = Field(default_factory=list)
    use_knowledge_bridge: bool = True
    mask: Optional[MaskEvidenceRequest] = None


class MaskEvidencePipelineResult(StrictModel):
    summary: MaskEvidenceSummary
    artifact: str


class EvidenceBundlePipelineResult(StrictModel):
    bundle: EvidenceBundle
    artifact: str


class RSAgentPipelineResult(StrictModel):
    run_id: str
    item_id: str
    plan: MainAgentPlan
    plan_artifact: str
    caption: Optional[RSCCPipelineResult] = None
    knowledge: Optional[KnowledgeBridgePacket] = None
    vqa: Optional[RSVQAPipelineResult] = None
    mask: Optional[MaskEvidencePipelineResult] = None
    evidence: EvidenceBundlePipelineResult
    result_artifact: str


class RSAgentPipeline:
    def __init__(
        self,
        cc_config: RSCCExperimentConfig,
        cc_providers: ProviderRegistry,
        vqa_config: RSVQAExperimentConfig,
        vqa_providers: ProviderRegistry,
        artifacts: JsonArtifactStore,
    ):
        self.artifacts = artifacts
        self.caption_pipeline = RSCCPipeline(cc_config, cc_providers, artifacts)
        self.vqa_pipeline = RSVQAPipeline(vqa_config, vqa_providers, artifacts)

    async def run(self, request: RSAgentRequest, run_id: str) -> RSAgentPipelineResult:
        plan = plan_request(
            request.task_type,
            use_knowledge_bridge=request.use_knowledge_bridge,
            has_mask=request.mask is not None,
        )
        plan_artifact = self._write_plan(plan, request, run_id)

        caption_result: Optional[RSCCPipelineResult] = None
        knowledge: Optional[KnowledgeBridgePacket] = None
        vqa_result: Optional[RSVQAPipelineResult] = None
        mask_result: Optional[MaskEvidencePipelineResult] = None

        if PipelineStage.RS_CC in plan.stages:
            caption_result = await self.caption_pipeline.run(
                RSCCRequest(
                    item_id=request.item_id,
                    original_caption=request.original_caption,
                ),
                run_id,
            )
        if PipelineStage.KNOWLEDGE_BRIDGE in plan.stages:
            if caption_result is None:
                raise RuntimeError("knowledge bridge requires a selected RS-CC caption")
            knowledge = bridge_selected_caption(caption_result)
        if PipelineStage.RS_VQA in plan.stages:
            vqa_result = await self.vqa_pipeline.run(
                RSVQARequest(
                    item_id=request.item_id,
                    images=request.images,
                    user_questions=request.user_questions,
                    knowledge_caption=knowledge.c_star if knowledge else None,
                ),
                run_id,
            )
        if request.mask is not None:
            mask_result = self._write_mask(analyze_mask(request.mask), request, run_id)

        evidence = self._write_evidence(
            build_evidence_bundle(
                original_caption=request.original_caption,
                caption=caption_result,
                vqa=vqa_result,
                mask=mask_result.summary if mask_result else None,
            ),
            request,
            run_id,
        )
        envelope = ArtifactEnvelope.create(
            artifact_type="rs_agent_result",
            run_id=run_id,
            item_id=request.item_id,
            payload={
                "item_id": request.item_id,
                "task_type": request.task_type.value,
                "plan": plan.model_dump(mode="json"),
                "original_caption": request.original_caption,
                "selected_caption": (
                    caption_result.selected_caption if caption_result else None
                ),
                "knowledge": knowledge.model_dump(mode="json") if knowledge else None,
                "selected_answers": (
                    [
                        answer.model_dump(mode="json")
                        for answer in vqa_result.selected_answers
                    ]
                    if vqa_result
                    else []
                ),
                "mask": (
                    mask_result.summary.model_dump(mode="json")
                    if mask_result
                    else None
                ),
                "evidence": evidence.bundle.model_dump(mode="json"),
                "source_artifacts": {
                    "main_agent_plan": str(plan_artifact),
                    "rs_cc_result": (
                        caption_result.result_artifact if caption_result else None
                    ),
                    "rs_vqa_result": vqa_result.result_artifact if vqa_result else None,
                    "mask_evidence": mask_result.artifact if mask_result else None,
                    "evidence_bundle": evidence.artifact,
                },
            },
        )
        result_artifact = self.artifacts.write(envelope)
        return RSAgentPipelineResult(
            run_id=run_id,
            item_id=request.item_id,
            plan=plan,
            plan_artifact=str(plan_artifact),
            caption=caption_result,
            knowledge=knowledge,
            vqa=vqa_result,
            mask=mask_result,
            evidence=evidence,
            result_artifact=str(result_artifact),
        )

    def _write_plan(
        self, plan: MainAgentPlan, request: RSAgentRequest, run_id: str
    ) -> Path:
        envelope = ArtifactEnvelope.create(
            artifact_type="main_agent_plan",
            run_id=run_id,
            item_id=request.item_id,
            payload={
                "plan": plan.model_dump(mode="json"),
                "request": {
                    "task_type": request.task_type.value,
                    "question_count": len(request.user_questions),
                    "use_knowledge_bridge": request.use_knowledge_bridge,
                    "has_mask": request.mask is not None,
                },
            },
        )
        return self.artifacts.write(envelope)

    def _write_mask(
        self, summary: MaskEvidenceSummary, request: RSAgentRequest, run_id: str
    ) -> MaskEvidencePipelineResult:
        envelope = ArtifactEnvelope.create(
            artifact_type="mask_evidence",
            run_id=run_id,
            item_id=request.item_id,
            payload=summary.model_dump(mode="json"),
            metadata={"source": summary.source.value},
        )
        artifact = self.artifacts.write(envelope)
        return MaskEvidencePipelineResult(summary=summary, artifact=str(artifact))

    def _write_evidence(
        self, bundle: EvidenceBundle, request: RSAgentRequest, run_id: str
    ) -> EvidenceBundlePipelineResult:
        envelope = ArtifactEnvelope.create(
            artifact_type="evidence_bundle",
            run_id=run_id,
            item_id=request.item_id,
            payload=bundle.model_dump(mode="json"),
            metadata={"layer": "engineering_enhancement"},
        )
        artifact = self.artifacts.write(envelope)
        return EvidenceBundlePipelineResult(bundle=bundle, artifact=str(artifact))
