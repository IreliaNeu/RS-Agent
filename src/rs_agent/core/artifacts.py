"""Immutable JSON artifact storage with integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from pydantic import Field

from rs_agent.core.schemas import StrictModel, utc_now


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(data: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


class ArtifactEnvelope(StrictModel):
    schema_version: str = "1.0"
    artifact_id: str
    artifact_type: str
    run_id: str
    item_id: Optional[str] = None
    created_at: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: str

    @classmethod
    def create(
        cls,
        artifact_type: str,
        run_id: str,
        payload: Dict[str, Any],
        item_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_id: Optional[str] = None,
    ) -> "ArtifactEnvelope":
        unsigned = {
            "schema_version": "1.0",
            "artifact_id": artifact_id or uuid4().hex,
            "artifact_type": artifact_type,
            "run_id": run_id,
            "item_id": item_id,
            "created_at": utc_now().isoformat(),
            "payload": payload,
            "metadata": metadata or {},
        }
        return cls(**unsigned, checksum=content_hash(unsigned))

    def verify(self) -> None:
        unsigned = self.model_dump(mode="json", exclude={"checksum"})
        actual = content_hash(unsigned)
        if actual != self.checksum:
            raise ValueError("artifact checksum mismatch: {}".format(self.artifact_id))


class JsonArtifactStore:
    """Write-once artifact store organized by run and artifact type."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def _safe_segment(value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("unsafe artifact path segment: {!r}".format(value))
        return value

    def path_for(self, artifact: ArtifactEnvelope) -> Path:
        run_id = self._safe_segment(artifact.run_id)
        artifact_type = self._safe_segment(artifact.artifact_type)
        artifact_id = self._safe_segment(artifact.artifact_id)
        return self.root / run_id / artifact_type / "{}.json".format(artifact_id)

    def write(self, artifact: ArtifactEnvelope) -> Path:
        artifact.verify()
        path = self.path_for(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            artifact.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def read(self, path: Path) -> ArtifactEnvelope:
        with path.open("r", encoding="utf-8") as handle:
            artifact = ArtifactEnvelope.model_validate(json.load(handle))
        artifact.verify()
        return artifact


def export_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

