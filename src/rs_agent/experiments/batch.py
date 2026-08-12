"""Batch orchestration with bounded concurrency and resumable state."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.experiments.cache import CachedResult, ResultCache
from rs_agent.experiments.identity import (
    BatchItem,
    ExperimentIdentity,
    canonical_sha256,
)
from rs_agent.experiments.state import (
    BatchRunState,
    BatchStateStore,
    BatchStatus,
    ItemStatus,
    finalize_status,
    utc_text,
)

RunItem = Callable[[BatchItem, str], Awaitable[str]]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def load_batch_items(path: Path) -> List[BatchItem]:
    base = path.resolve().parent
    items = []
    known = {
        "item_id",
        "original_caption",
        "image_a",
        "image_b",
        "predicted_mask",
        "user_questions",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on input line {}: {}".format(line_number, exc))
            if not isinstance(data, dict):
                raise ValueError("input line {} must contain a JSON object".format(line_number))
            for field in ("image_a", "image_b", "predicted_mask"):
                value = data.get(field)
                if value and not Path(value).is_absolute():
                    data[field] = base / value
            data["metadata"] = {key: value for key, value in data.items() if key not in known}
            try:
                item = BatchItem.model_validate({key: value for key, value in data.items() if key in known or key == "metadata"})
            except ValueError as exc:
                raise ValueError("invalid input line {}: {}".format(line_number, exc))
            if not item.original_caption.strip():
                raise ValueError("input line {} has an empty original_caption".format(line_number))
            items.append(item)
    if not items:
        raise ValueError("batch input is empty")
    return items


def cache_key_for(experiment: ExperimentIdentity, item_id: str) -> str:
    return canonical_sha256(
        {
            "source_sha256": experiment.source_sha256,
            "runtime": experiment.runtime,
            "config_sha256": experiment.config_sha256,
            "options": experiment.options,
            "item_fingerprint": experiment.item_fingerprints[item_id],
        }
    )


def _result_is_valid(path: Optional[str], item_id: str) -> bool:
    if not path:
        return False
    try:
        artifact = JsonArtifactStore(Path(".")).read(Path(path))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return artifact.artifact_type == "rs_agent_result" and artifact.item_id == item_id


class BatchRunner:
    def __init__(
        self,
        *,
        batch_id: str,
        items: Iterable[BatchItem],
        experiment: ExperimentIdentity,
        state_store: BatchStateStore,
        run_item: RunItem,
        concurrency: int = 1,
        retry_failed: bool = False,
        max_items: Optional[int] = None,
        result_cache: Optional[ResultCache] = None,
    ):
        if not SAFE_ID.fullmatch(batch_id):
            raise ValueError("batch_id must contain only letters, digits, dot, dash, or underscore")
        if concurrency < 1:
            raise ValueError("concurrency must be at least one")
        if max_items is not None and max_items < 1:
            raise ValueError("max_items must be at least one")
        self.batch_id = batch_id
        self.items = {item.item_id: item for item in items}
        self.experiment = experiment
        self.state_store = state_store
        self.run_item = run_item
        self.concurrency = concurrency
        self.retry_failed = retry_failed
        self.max_items = max_items
        self.result_cache = result_cache
        self._state_lock: Optional[asyncio.Lock] = None

    async def run(self) -> BatchRunState:
        self._state_lock = asyncio.Lock()
        state = self.state_store.create_or_load(
            batch_id=self.batch_id,
            experiment=self.experiment,
        )
        state = self._repair_interrupted_or_invalid(state)
        state = state.model_copy(update={"status": BatchStatus.RUNNING})
        self.state_store.write(state)

        eligible = [
            item_id
            for item_id, item_state in state.items.items()
            if item_state.status == ItemStatus.PENDING
            or (self.retry_failed and item_state.status == ItemStatus.FAILED)
        ]
        if self.max_items is not None:
            eligible = eligible[: self.max_items]
        semaphore = asyncio.Semaphore(self.concurrency)

        assert self._state_lock is not None
        async def guarded(item_id: str) -> None:
            async with semaphore:
                await self._run_one(item_id)

        await asyncio.gather(*(guarded(item_id) for item_id in eligible))
        async with self._state_lock:
            latest = self.state_store.read()
            latest = latest.model_copy(update={"status": finalize_status(latest)})
            self.state_store.write(latest)
            return self.state_store.read()

    def _repair_interrupted_or_invalid(self, state: BatchRunState) -> BatchRunState:
        updates = dict(state.items)
        changed = False
        for item_id, item_state in state.items.items():
            if item_state.status == ItemStatus.COMPLETED:
                if _result_is_valid(item_state.result_artifact, item_id):
                    continue
                updates[item_id] = item_state.model_copy(
                    update={
                        "status": ItemStatus.PENDING,
                        "result_artifact": None,
                        "error": "completed result was missing or failed integrity validation",
                    }
                )
                changed = True
            elif item_state.status == ItemStatus.RUNNING:
                updates[item_id] = item_state.model_copy(
                    update={
                        "status": ItemStatus.PENDING,
                        "error": "previous invocation ended while this item was running",
                    }
                )
                changed = True
        repaired = state.model_copy(update={"items": updates}) if changed else state
        if changed:
            self.state_store.write(repaired)
        return repaired

    async def _run_one(self, item_id: str) -> None:
        item = self.items[item_id]
        cache_key = cache_key_for(self.experiment, item_id)
        assert self._state_lock is not None
        if self.result_cache is not None:
            cached = self.result_cache.lookup(cache_key, item_id)
            if cached is not None:
                await self._mark_cached(item_id, cached)
                return

        async with self._state_lock:
            state = self.state_store.read()
            previous = state.items[item_id]
            attempt = previous.attempts + 1
            run_id = "{}-{}-a{:02d}".format(
                self.batch_id,
                re.sub(r"[^A-Za-z0-9._-]", "_", item_id)[:64],
                attempt,
            )
            updated_item = previous.model_copy(
                update={
                    "status": ItemStatus.RUNNING,
                    "attempts": attempt,
                    "run_id": run_id,
                    "result_artifact": None,
                    "cache_hit": False,
                    "error": None,
                    "started_at": utc_text(),
                    "completed_at": None,
                }
            )
            items = dict(state.items)
            items[item_id] = updated_item
            self.state_store.write(state.model_copy(update={"items": items}))

        try:
            result_artifact = await self.run_item(item, run_id)
            if not _result_is_valid(result_artifact, item_id):
                raise RuntimeError("pipeline returned an invalid result artifact")
        except Exception as exc:
            await self._finish(
                item_id,
                status=ItemStatus.FAILED,
                error="{}: {}".format(type(exc).__name__, exc)[:2000],
            )
            return

        await self._finish(
            item_id,
            status=ItemStatus.COMPLETED,
            result_artifact=result_artifact,
        )
        if self.result_cache is not None:
            state = self.state_store.read()
            completed = state.items[item_id]
            self.result_cache.store(
                CachedResult(
                    cache_key=cache_key,
                    item_id=item_id,
                    run_id=completed.run_id or "",
                    result_artifact=result_artifact,
                )
            )

    async def _mark_cached(self, item_id: str, cached: CachedResult) -> None:
        assert self._state_lock is not None
        async with self._state_lock:
            state = self.state_store.read()
            previous = state.items[item_id]
            items = dict(state.items)
            items[item_id] = previous.model_copy(
                update={
                    "status": ItemStatus.COMPLETED,
                    "run_id": cached.run_id,
                    "result_artifact": cached.result_artifact,
                    "cache_hit": True,
                    "error": None,
                    "completed_at": utc_text(),
                }
            )
            self.state_store.write(state.model_copy(update={"items": items}))

    async def _finish(
        self,
        item_id: str,
        *,
        status: ItemStatus,
        result_artifact: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        assert self._state_lock is not None
        async with self._state_lock:
            state = self.state_store.read()
            previous = state.items[item_id]
            items = dict(state.items)
            items[item_id] = previous.model_copy(
                update={
                    "status": status,
                    "result_artifact": result_artifact,
                    "error": error,
                    "completed_at": utc_text(),
                }
            )
            self.state_store.write(state.model_copy(update={"items": items}))
