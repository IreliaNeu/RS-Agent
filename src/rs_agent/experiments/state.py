"""Atomic, resumable state for batch orchestration."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from pydantic import Field

from rs_agent.core.schemas import StrictModel
from rs_agent.experiments.identity import ExperimentIdentity


def utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


class ItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    PARTIAL = "partial"


class BatchItemState(StrictModel):
    item_id: str
    item_fingerprint: str
    status: ItemStatus = ItemStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    cache_hit: bool = False
    run_id: Optional[str] = None
    result_artifact: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class BatchRunState(StrictModel):
    schema_version: str = "1.0"
    batch_id: str
    experiment: ExperimentIdentity
    status: BatchStatus = BatchStatus.PENDING
    created_at: str = Field(default_factory=utc_text)
    updated_at: str = Field(default_factory=utc_text)
    items: Dict[str, BatchItemState]

    def counts(self) -> Dict[str, int]:
        return {
            status.value: sum(item.status == status for item in self.items.values())
            for status in ItemStatus
        }


class BatchStateStore:
    def __init__(self, path: Path):
        self.path = path

    def create_or_load(
        self,
        *,
        batch_id: str,
        experiment: ExperimentIdentity,
    ) -> BatchRunState:
        if self.path.exists():
            state = self.read()
            if state.batch_id != batch_id:
                raise ValueError("batch state belongs to a different batch_id")
            if state.experiment.fingerprint != experiment.fingerprint:
                raise ValueError(
                    "batch_id already exists with different input, configs, source, runtime, "
                    "or options; choose a new batch_id"
                )
            return state
        state = BatchRunState(
            batch_id=batch_id,
            experiment=experiment,
            items={
                item_id: BatchItemState(
                    item_id=item_id,
                    item_fingerprint=fingerprint,
                )
                for item_id, fingerprint in experiment.item_fingerprints.items()
            },
        )
        self.write(state)
        return self.read()

    def read(self) -> BatchRunState:
        with self.path.open("r", encoding="utf-8") as handle:
            return BatchRunState.model_validate(json.load(handle))

    def write(self, state: BatchRunState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        updated = state.model_copy(update={"updated_at": utc_text()})
        temporary = self.path.with_name("{}.tmp".format(self.path.name))
        serialized = json.dumps(updated.model_dump(mode="json"), ensure_ascii=False, indent=2)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


def finalize_status(state: BatchRunState) -> BatchStatus:
    statuses = {item.status for item in state.items.values()}
    if statuses == {ItemStatus.COMPLETED}:
        return BatchStatus.COMPLETED
    if ItemStatus.PENDING not in statuses and ItemStatus.RUNNING not in statuses:
        return BatchStatus.COMPLETED_WITH_ERRORS
    return BatchStatus.PARTIAL
