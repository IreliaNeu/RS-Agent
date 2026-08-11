from rs_agent.core.schemas import TaskType
from rs_agent.orchestration.main_agent import PipelineStage, plan_request


def test_caption_request_runs_only_rs_cc() -> None:
    plan = plan_request(
        TaskType.CAPTION_ENRICHMENT,
        use_knowledge_bridge=True,
        has_mask=False,
    )
    assert plan.stages == [PipelineStage.RS_CC]
    assert plan.return_caption is True
    assert plan.return_answers is False
    assert plan.use_knowledge_bridge is False


def test_vqa_request_with_bridge_generates_c_star_first() -> None:
    plan = plan_request(TaskType.VQA, use_knowledge_bridge=True, has_mask=False)
    assert plan.stages == [
        PipelineStage.RS_CC,
        PipelineStage.KNOWLEDGE_BRIDGE,
        PipelineStage.RS_VQA,
    ]
    assert plan.return_caption is False
    assert plan.return_answers is True


def test_vqa_ablation_skips_rs_cc_and_bridge() -> None:
    plan = plan_request(TaskType.VQA, use_knowledge_bridge=False, has_mask=True)
    assert plan.stages == [PipelineStage.RS_VQA, PipelineStage.MASK_EVIDENCE]
    assert plan.use_knowledge_bridge is False
    assert plan.include_mask is True
