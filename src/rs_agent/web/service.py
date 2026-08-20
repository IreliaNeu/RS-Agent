"""Application service and presentation view for the Streamlit demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ImagePair, StrictModel, TaskType
from rs_agent.domains.remote_sensing.config import load_rs_cc_config
from rs_agent.domains.remote_sensing.mask_evidence import MaskEvidenceRequest, MaskSource
from rs_agent.domains.remote_sensing.vqa_config import load_rs_vqa_config
from rs_agent.orchestration.agent_pipeline import (
    RSAgentPipeline,
    RSAgentPipelineResult,
    RSAgentRequest,
)
from rs_agent.providers.registry import ProviderRegistry


class DemoPipelineRequest(StrictModel):
    run_id: str
    item_id: str
    image_a: Path
    image_b: Path
    original_caption: str
    task_type: TaskType
    questions: List[str] = Field(default_factory=list)
    use_knowledge_bridge: bool = True
    mask: Optional[Path] = None
    mask_source: MaskSource = MaskSource.PREDICTED
    mask_min_component_pixels: int = Field(default=1, ge=1)
    mask_min_changed_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class CaptionCandidateView(StrictModel):
    label: str
    model_name: str
    provider: str
    model_id: str
    input_mode: str
    generation_mode: str
    text: str
    score: Optional[int] = None
    selected: bool = False
    judge_choice: bool = False
    error: Optional[str] = None


class VQACandidateView(StrictModel):
    question_id: str
    question: str
    label: str
    model_name: str
    text: str
    score: Optional[int] = None
    selected: bool = False
    judge_choice: bool = False
    error: Optional[str] = None


class SelectedAnswerView(StrictModel):
    question: str
    answer: str
    model_name: str


class DemoRunView(StrictModel):
    run_id: str
    item_id: str
    stages: List[str]
    selected_caption: Optional[str] = None
    selected_caption_label: Optional[str] = None
    knowledge_caption: Optional[str] = None
    caption_candidates: List[CaptionCandidateView] = Field(default_factory=list)
    selected_answers: List[SelectedAnswerView] = Field(default_factory=list)
    vqa_candidates: List[VQACandidateView] = Field(default_factory=list)
    mask_summary: Optional[Dict[str, Any]] = None
    evidence: Dict[str, Any]
    artifacts: Dict[str, str]


def _read_payload(path: str, expected_type: str) -> Dict[str, Any]:
    artifact = JsonArtifactStore(Path(".")).read(Path(path))
    if artifact.artifact_type != expected_type:
        raise ValueError(
            "expected {} artifact, got {}".format(
                expected_type, artifact.artifact_type
            )
        )
    return artifact.payload


def build_demo_view(result: RSAgentPipelineResult) -> DemoRunView:
    caption_candidates: List[CaptionCandidateView] = []
    vqa_candidates: List[VQACandidateView] = []
    selected_answers: List[SelectedAnswerView] = []
    artifacts = {
        "main_agent_plan": result.plan_artifact,
        "evidence_bundle": result.evidence.artifact,
        "rs_agent_result": result.result_artifact,
    }
    if result.caption is not None:
        generation = _read_payload(
            result.caption.generation_artifact, "caption_candidates"
        )
        evaluation = _read_payload(
            result.caption.evaluation_artifact, "caption_evaluation"
        )
        scores = evaluation.get("scores", {})
        selected_label = evaluation["selection"]["selected_label"]
        judge_choice = evaluation.get("judge_choice")
        for candidate in generation.get("candidates", []):
            caption_candidates.append(
                CaptionCandidateView(
                    label=candidate["label"],
                    model_name=candidate["model"]["name"],
                    provider=candidate["model"]["provider"],
                    model_id=candidate["model"]["model"],
                    input_mode=candidate.get("input_mode", "text_only"),
                    generation_mode=candidate.get("generation_mode", "generated"),
                    text=candidate.get("text", ""),
                    score=scores.get(candidate["label"]),
                    selected=candidate["label"] == selected_label,
                    judge_choice=candidate["label"] == judge_choice,
                    error=candidate.get("error"),
                )
            )
        artifacts.update(
            {
                "caption_candidates": result.caption.generation_artifact,
                "caption_evaluation": result.caption.evaluation_artifact,
                "rs_cc_result": result.caption.result_artifact,
            }
        )
    if result.vqa is not None:
        generation = _read_payload(result.vqa.generation_artifact, "vqa_candidates")
        evaluation = _read_payload(result.vqa.evaluation_artifact, "vqa_evaluation")
        questions = {
            question["question_id"]: question["text"]
            for question in generation.get("questions", [])
        }
        records = {
            record["question"]["question_id"]: record
            for record in evaluation.get("records", [])
        }
        for candidate in generation.get("candidates", []):
            record = records.get(candidate["question_id"], {})
            selection = record.get("selection", {})
            vqa_candidates.append(
                VQACandidateView(
                    question_id=candidate["question_id"],
                    question=questions.get(candidate["question_id"], ""),
                    label=candidate["label"],
                    model_name=candidate["model"]["name"],
                    text=candidate.get("text", ""),
                    score=record.get("scores", {}).get(candidate["label"]),
                    selected=candidate["label"] == selection.get("selected_label"),
                    judge_choice=candidate["label"] == record.get("judge_choice"),
                    error=candidate.get("error"),
                )
            )
        selected_answers = [
            SelectedAnswerView(
                question=answer.question.text,
                answer=answer.answer,
                model_name=answer.model_name,
            )
            for answer in result.vqa.selected_answers
        ]
        artifacts.update(
            {
                "vqa_candidates": result.vqa.generation_artifact,
                "vqa_evaluation": result.vqa.evaluation_artifact,
                "rs_vqa_result": result.vqa.result_artifact,
            }
        )
    if result.mask is not None:
        artifacts["mask_evidence"] = result.mask.artifact
    return DemoRunView(
        run_id=result.run_id,
        item_id=result.item_id,
        stages=[stage.value for stage in result.plan.stages],
        selected_caption=(result.caption.selected_caption if result.caption else None),
        selected_caption_label=(result.caption.selected_label if result.caption else None),
        knowledge_caption=(result.knowledge.c_star if result.knowledge else None),
        caption_candidates=caption_candidates,
        selected_answers=selected_answers,
        vqa_candidates=vqa_candidates,
        mask_summary=(
            result.mask.summary.model_dump(mode="json") if result.mask else None
        ),
        evidence=result.evidence.bundle.model_dump(mode="json"),
        artifacts=artifacts,
    )


async def run_demo_pipeline(
    request: DemoPipelineRequest,
    *,
    cc_config_path: Path,
    vqa_config_path: Path,
    artifact_dir: Path,
) -> DemoRunView:
    cc_config = load_rs_cc_config(cc_config_path)
    vqa_config = load_rs_vqa_config(vqa_config_path)
    cc_registry = ProviderRegistry(cc_config.model_registry())
    vqa_registry = ProviderRegistry(vqa_config.provider_registry())
    try:
        pipeline = RSAgentPipeline(
            cc_config,
            cc_registry,
            vqa_config,
            vqa_registry,
            JsonArtifactStore(artifact_dir),
        )
        result = await pipeline.run(
            RSAgentRequest(
                item_id=request.item_id,
                original_caption=request.original_caption,
                images=ImagePair(before=request.image_a, after=request.image_b),
                task_type=request.task_type,
                user_questions=request.questions,
                use_knowledge_bridge=request.use_knowledge_bridge,
                mask=(
                    MaskEvidenceRequest(
                        path=request.mask,
                        source=request.mask_source,
                        min_component_pixels=request.mask_min_component_pixels,
                        min_changed_ratio=request.mask_min_changed_ratio,
                    )
                    if request.mask is not None
                    else None
                ),
            ),
            request.run_id,
        )
        return build_demo_view(result)
    finally:
        await cc_registry.close()
        await vqa_registry.close()
