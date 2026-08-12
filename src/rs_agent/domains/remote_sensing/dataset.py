"""Input records for Change-Agent captions evaluated on LEVIR-MCI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from pydantic import Field, model_validator

from rs_agent.core.schemas import StrictModel


class RSCCInputRecord(StrictModel):
    item_id: str
    original_caption: str
    image_a: Optional[Path] = None
    image_b: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> "RSCCInputRecord":
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if not self.original_caption.strip():
            raise ValueError("original_caption cannot be empty")
        if (self.image_a is None) != (self.image_b is None):
            raise ValueError("image_a and image_b must be supplied together")
        return self


def parse_rs_cc_record(data: Dict[str, Any], line_number: int) -> RSCCInputRecord:
    """Accept normalized records and the legacy evaluation JSONL field names."""

    item_id = data.get("item_id") or data.get("image_id") or data.get("Image ID")
    caption = data.get("original_caption") or data.get("Original Caption")
    if item_id is None:
        item_id = "item_{:06d}".format(line_number)
    if not isinstance(item_id, (str, int)):
        raise ValueError("line {} has an invalid item_id".format(line_number))
    if not isinstance(caption, str):
        raise ValueError(
            "line {} must contain original_caption or Original Caption".format(line_number)
        )
    excluded = {
        "item_id",
        "image_id",
        "Image ID",
        "original_caption",
        "Original Caption",
        "image_a",
        "image_b",
    }
    metadata = {key: value for key, value in data.items() if key not in excluded}
    return RSCCInputRecord(
        item_id=str(item_id).strip(),
        original_caption=caption.strip(),
        image_a=data.get("image_a"),
        image_b=data.get("image_b"),
        metadata=metadata,
    )


def iter_rs_cc_jsonl(path: Path, limit: Optional[int] = None) -> Iterator[RSCCInputRecord]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    base = path.resolve().parent
    with path.open("r", encoding="utf-8") as handle:
        emitted = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on line {}: {}".format(line_number, exc)) from exc
            if not isinstance(data, dict):
                raise ValueError("line {} must be a JSON object".format(line_number))
            for field in ("image_a", "image_b"):
                value = data.get(field)
                if value and not Path(value).is_absolute():
                    data[field] = base / value
            yield parse_rs_cc_record(data, line_number)
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def materialize_records(records: Iterable[RSCCInputRecord]) -> List[RSCCInputRecord]:
    materialized = list(records)
    if not materialized:
        raise ValueError("RS-CC input contains no records")
    return materialized
