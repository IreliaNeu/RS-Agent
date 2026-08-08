"""End-to-end RS-CC, Knowledge Bridge, and RS-VQA orchestration."""

from __future__ import annotations

from typing import List

from pydantic import Field

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.core.schemas import ImagePair, StrictModel
from rs_agent.domains.remote_sensing.caption_agent import RSCCRequest
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.domains.remote_sensing.vqa_agent import RSVQARequest
from rs_agent.domains.remote_sensing.vqa_config import RSVQAExperimentConfig
from rs_agent.orchestration.caption_pipeline import RSCCPipeline, RSCCPipelineResult
from rs_agent.orchestration.knowledge_bridge import (
    KnowledgeBridgePacket,
    bridge_selected_caption,
)
from rs_agent.orchestration.vqa_pipeline import RSVQAPipeline, RSVQAPipelineResult
from rs_agent.providers.registry import ProviderRegistry


class RSAgentRequest(StrictModel):
    item_id: str
    original_caption: str
    images: ImagePair
    user_questions: List[str] = Field(default_factory=list)


class RSAgentPipelineResult(StrictModel):
    run_id: str
    item_id: str
    caption: RSCCPipelineResult
    knowledge: KnowledgeBridgePacket
    vqa: RSVQAPipelineResult
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
        caption_result = await self.caption_pipeline.run(
            RSCCRequest(
                item_id=request.item_id,
                original_caption=request.original_caption,
            ),
            run_id,
        )
        knowledge = bridge_selected_caption(caption_result)
        vqa_result = await self.vqa_pipeline.run(
            RSVQARequest(
                item_id=request.item_id,
                images=request.images,
                user_questions=request.user_questions,
                knowledge_caption=knowledge.c_star,
            ),
            run_id,
        )
        envelope = ArtifactEnvelope.create(
            artifact_type="rs_agent_result",
            run_id=run_id,
            item_id=request.item_id,
            payload={
                "item_id": request.item_id,
                "original_caption": request.original_caption,
                "knowledge": knowledge.model_dump(mode="json"),
                "selected_answers": [
                    answer.model_dump(mode="json")
                    for answer in vqa_result.selected_answers
                ],
                "source_artifacts": {
                    "rs_cc_result": caption_result.result_artifact,
                    "rs_vqa_result": vqa_result.result_artifact,
                },
            },
        )
        result_artifact = self.artifacts.write(envelope)
        return RSAgentPipelineResult(
            run_id=run_id,
            item_id=request.item_id,
            caption=caption_result,
            knowledge=knowledge,
            vqa=vqa_result,
            result_artifact=str(result_artifact),
        )
