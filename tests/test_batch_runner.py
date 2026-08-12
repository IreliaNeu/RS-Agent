import asyncio
from pathlib import Path

from PIL import Image

from rs_agent.core.artifacts import ArtifactEnvelope, JsonArtifactStore
from rs_agent.experiments.batch import BatchRunner, cache_key_for
from rs_agent.experiments.cache import ResultCache
from rs_agent.experiments.identity import BatchItem, build_experiment_identity
from rs_agent.experiments.state import BatchStateStore, BatchStatus, ItemStatus


def prepare(tmp_path: Path, count: int = 3):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(before)
    Image.new("RGB", (2, 2), color=(3, 2, 1)).save(after)
    items = [
        BatchItem(
            item_id="item-{}".format(index),
            original_caption="No change.",
            image_a=before,
            image_b=after,
        )
        for index in range(count)
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("profile: test\n", encoding="utf-8")
    experiment = build_experiment_identity(
        input_path=input_path,
        config_paths={"config": config},
        options={"task": "caption"},
        items=items,
    )
    return items, experiment


def artifact_callback(root: Path, calls: list, failures=None):
    failures = failures or set()

    async def run(item: BatchItem, run_id: str) -> str:
        calls.append(item.item_id)
        if item.item_id in failures:
            raise RuntimeError("planned failure")
        artifact = ArtifactEnvelope.create(
            artifact_type="rs_agent_result",
            run_id=run_id,
            item_id=item.item_id,
            payload={"item_id": item.item_id},
        )
        return str(JsonArtifactStore(root).write(artifact))

    return run


def test_batch_resumes_pending_items_without_rerunning_completed(tmp_path: Path) -> None:
    items, experiment = prepare(tmp_path)
    state_store = BatchStateStore(tmp_path / "state.json")
    calls = []
    first = BatchRunner(
        batch_id="resume-test",
        items=items,
        experiment=experiment,
        state_store=state_store,
        run_item=artifact_callback(tmp_path / "artifacts", calls),
        concurrency=2,
        max_items=1,
    )
    partial = asyncio.run(first.run())

    assert partial.status == BatchStatus.PARTIAL
    assert partial.counts() == {"pending": 2, "running": 0, "completed": 1, "failed": 0}

    second = BatchRunner(
        batch_id="resume-test",
        items=items,
        experiment=experiment,
        state_store=state_store,
        run_item=artifact_callback(tmp_path / "artifacts", calls),
        concurrency=2,
    )
    completed = asyncio.run(second.run())

    assert completed.status == BatchStatus.COMPLETED
    assert sorted(calls) == ["item-0", "item-1", "item-2"]
    assert all(item.attempts == 1 for item in completed.items.values())


def test_failed_items_retry_only_when_requested(tmp_path: Path) -> None:
    items, experiment = prepare(tmp_path, count=1)
    state_store = BatchStateStore(tmp_path / "state.json")
    failed = BatchRunner(
        batch_id="retry-test",
        items=items,
        experiment=experiment,
        state_store=state_store,
        run_item=artifact_callback(tmp_path / "artifacts", [], {"item-0"}),
    )
    state = asyncio.run(failed.run())
    assert state.status == BatchStatus.COMPLETED_WITH_ERRORS
    assert state.items["item-0"].status == ItemStatus.FAILED

    calls = []
    retried = BatchRunner(
        batch_id="retry-test",
        items=items,
        experiment=experiment,
        state_store=state_store,
        run_item=artifact_callback(tmp_path / "artifacts", calls),
        retry_failed=True,
    )
    state = asyncio.run(retried.run())
    assert state.status == BatchStatus.COMPLETED
    assert state.items["item-0"].attempts == 2
    assert calls == ["item-0"]


def test_cache_reuses_verified_result_across_batch_ids(tmp_path: Path) -> None:
    items, experiment = prepare(tmp_path, count=1)
    cache = ResultCache(tmp_path / "cache")
    calls = []
    first = BatchRunner(
        batch_id="cache-one",
        items=items,
        experiment=experiment,
        state_store=BatchStateStore(tmp_path / "one.json"),
        run_item=artifact_callback(tmp_path / "artifacts", calls),
        result_cache=cache,
    )
    asyncio.run(first.run())
    second = BatchRunner(
        batch_id="cache-two",
        items=items,
        experiment=experiment,
        state_store=BatchStateStore(tmp_path / "two.json"),
        run_item=artifact_callback(tmp_path / "artifacts", calls),
        result_cache=cache,
    )
    state = asyncio.run(second.run())

    assert calls == ["item-0"]
    assert state.items["item-0"].attempts == 0
    assert state.items["item-0"].cache_hit is True
    assert state.items["item-0"].error is None


def test_cache_key_changes_when_source_changes(tmp_path: Path) -> None:
    _, experiment = prepare(tmp_path, count=1)
    changed = experiment.model_copy(update={"source_sha256": "0" * 64})

    assert cache_key_for(experiment, "item-0") != cache_key_for(changed, "item-0")
