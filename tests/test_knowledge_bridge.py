from rs_agent.orchestration.caption_pipeline import RSCCPipelineResult
from rs_agent.orchestration.knowledge_bridge import bridge_selected_caption


def test_knowledge_bridge_exposes_only_c_star_and_provenance() -> None:
    result = RSCCPipelineResult(
        run_id="run-1",
        request_id="request-1",
        item_id="pair-1",
        selected_label="D",
        selected_caption="Two buildings appeared near the central road.",
        generation_artifact="artifacts/candidates.json",
        evaluation_artifact="artifacts/evaluation.json",
        result_artifact="artifacts/result.json",
        candidate_count=5,
    )
    packet = bridge_selected_caption(result)
    assert packet.model_dump() == {
        "run_id": "run-1",
        "item_id": "pair-1",
        "c_star": "Two buildings appeared near the central road.",
        "source_result_artifact": "artifacts/result.json",
    }
