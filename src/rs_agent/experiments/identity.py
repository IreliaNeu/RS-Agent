"""Content-derived experiment identities for reproducible batch runs."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field

from rs_agent.core.schemas import StrictModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def runtime_identity() -> Dict[str, str]:
    identity = {"python": platform.python_version()}
    for distribution in ("httpx", "pillow", "pydantic", "pyyaml"):
        try:
            identity[distribution] = version(distribution)
        except PackageNotFoundError:
            identity[distribution] = "not-installed"
    return identity


def source_tree_sha256(root: Optional[Path] = None) -> str:
    package_root = root or Path(__file__).resolve().parents[1]
    entries = []
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        entries.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return canonical_sha256(entries)


class BatchItem(StrictModel):
    item_id: str
    original_caption: str
    image_a: Path
    image_b: Path
    predicted_mask: Optional[Path] = None
    user_questions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def content_identity(self) -> Dict[str, Any]:
        paths = {
            "image_a": self.image_a,
            "image_b": self.image_b,
            "predicted_mask": self.predicted_mask,
        }
        files: Dict[str, Any] = {}
        for name, path in paths.items():
            if path is None:
                files[name] = None
                continue
            resolved = path.resolve()
            if not resolved.is_file():
                raise ValueError("{} file does not exist: {}".format(name, resolved))
            files[name] = {"sha256": sha256_file(resolved)}
        return {
            "item_id": self.item_id,
            "original_caption": self.original_caption,
            "files": files,
            "user_questions": self.user_questions,
            "metadata": self.metadata,
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.content_identity())


class ExperimentIdentity(StrictModel):
    fingerprint: str
    input_sha256: str
    source_sha256: str
    runtime: Dict[str, str] = Field(default_factory=runtime_identity)
    config_sha256: Dict[str, str]
    options: Dict[str, Any]
    item_fingerprints: Dict[str, str]


def build_experiment_identity(
    *,
    input_path: Path,
    config_paths: Dict[str, Path],
    options: Dict[str, Any],
    items: Iterable[BatchItem],
    source_sha256: Optional[str] = None,
) -> ExperimentIdentity:
    materialized = list(items)
    item_ids = [item.item_id for item in materialized]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("batch input item_id values must be unique")
    item_fingerprints = {item.item_id: item.fingerprint() for item in materialized}
    config_sha256 = {
        name: sha256_file(path.resolve()) for name, path in sorted(config_paths.items())
    }
    input_sha256 = sha256_file(input_path.resolve())
    source_identity = source_sha256 or source_tree_sha256()
    runtime = runtime_identity()
    payload = {
        "input_sha256": input_sha256,
        "source_sha256": source_identity,
        "runtime": runtime,
        "config_sha256": config_sha256,
        "options": options,
        "item_fingerprints": item_fingerprints,
    }
    return ExperimentIdentity(
        fingerprint=canonical_sha256(payload),
        input_sha256=input_sha256,
        source_sha256=source_identity,
        runtime=runtime,
        config_sha256=config_sha256,
        options=options,
        item_fingerprints=item_fingerprints,
    )
