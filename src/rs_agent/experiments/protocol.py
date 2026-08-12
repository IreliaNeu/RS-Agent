"""Explicit experiment tracks and method variants for auditable comparisons."""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from rs_agent.core.schemas import StrictModel


class ExperimentTrack(str, Enum):
    PAPER = "paper"
    OPERATIONAL = "operational"
    ENHANCEMENT = "enhancement"


class ExperimentProtocol(StrictModel):
    track: ExperimentTrack
    baseline_id: str
    method_variant: str
    substitutes_paper_models: bool = False

    @model_validator(mode="after")
    def validate_track(self) -> "ExperimentProtocol":
        values = (self.baseline_id, self.method_variant)
        if any(not value.strip() for value in values):
            raise ValueError("baseline_id and method_variant cannot be empty")
        if self.track == ExperimentTrack.PAPER and self.substitutes_paper_models:
            raise ValueError("paper track cannot substitute paper models")
        return self
