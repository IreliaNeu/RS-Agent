"""Minimal knowledge handoff from RS-CC to downstream reasoning stages."""

from __future__ import annotations

from pydantic import model_validator

from rs_agent.core.schemas import StrictModel
from rs_agent.orchestration.caption_pipeline import RSCCPipelineResult


class KnowledgeBridgePacket(StrictModel):
    run_id: str
    item_id: str
    c_star: str
    source_result_artifact: str

    @model_validator(mode="after")
    def validate_caption(self) -> "KnowledgeBridgePacket":
        if not self.c_star.strip():
            raise ValueError("Knowledge Bridge requires a non-empty selected caption C*")
        return self


def bridge_selected_caption(result: RSCCPipelineResult) -> KnowledgeBridgePacket:
    """Expose C* and its provenance, never the rejected candidate set."""
    return KnowledgeBridgePacket(
        run_id=result.run_id,
        item_id=result.item_id,
        c_star=result.selected_caption,
        source_result_artifact=result.result_artifact,
    )
