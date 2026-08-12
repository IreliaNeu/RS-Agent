"""Opt-in cross-batch cache for fully completed, checksum-verified results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import StrictModel


class CachedResult(StrictModel):
    cache_key: str
    item_id: str
    run_id: str
    result_artifact: str


class ResultCache:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, cache_key: str) -> Path:
        if len(cache_key) != 64 or any(
            character not in "0123456789abcdef" for character in cache_key
        ):
            raise ValueError("invalid cache key")
        return self.root / "{}.json".format(cache_key)

    def lookup(self, cache_key: str, item_id: str) -> Optional[CachedResult]:
        path = self.path_for(cache_key)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = CachedResult.model_validate(json.load(handle))
            if record.cache_key != cache_key or record.item_id != item_id:
                return None
            artifact = JsonArtifactStore(Path(".")).read(Path(record.result_artifact))
            if (
                artifact.artifact_type != "rs_agent_result"
                or artifact.item_id != item_id
                or artifact.run_id != record.run_id
            ):
                return None
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return record

    def store(self, record: CachedResult) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(record.cache_key)
        if path.exists():
            return
        temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
        serialized = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2)
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
