"""Compatibility exports for the original evaluation JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from rs_agent.core.artifacts import export_jsonl
from rs_agent.evaluation.caption_judge import CaptionEvaluationRecord


def export_caption_evaluations(
    records: Iterable[CaptionEvaluationRecord],
    full_output: Path,
    best_output: Path,
    mapping_output: Path,
) -> None:
    materialized = list(records)
    export_jsonl(full_output, (record.legacy_full_record() for record in materialized))
    export_jsonl(best_output, (record.legacy_best_record() for record in materialized))

    mapping: Dict[str, str] = {}
    for record in materialized:
        for candidate in record.candidates:
            mapping[candidate.label] = candidate.model.name
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    with mapping_output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

