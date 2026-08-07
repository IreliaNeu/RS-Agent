"""Strict parsing and deterministic caption selection utilities."""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional

from rs_agent.core.schemas import (
    CandidateScore,
    CaptionCandidate,
    CaptionSelection,
)


def parse_choice(content: str, valid_labels: Iterable[str]) -> Optional[str]:
    labels = {label.upper() for label in valid_labels}
    normalized = content.strip().upper()
    if normalized in labels:
        return normalized
    for match in re.finditer(r"\b([A-Z])\b", normalized):
        if match.group(1) in labels:
            return match.group(1)
    return None


def parse_scores(content: str, labels: Iterable[str]) -> Dict[str, Optional[int]]:
    ordered_labels = [label.upper() for label in labels]
    result: Dict[str, Optional[int]] = {label: None for label in ordered_labels}
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return result
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return result
    if not isinstance(raw, dict):
        return result
    for key, value in raw.items():
        label = str(key).strip().upper()
        if label not in result or isinstance(value, bool):
            continue
        try:
            score = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= score <= 10:
            result[label] = score
    return result


def select_caption(
    candidates: List[CaptionCandidate],
    scores: Dict[str, Optional[int]],
    judge_choice: Optional[str],
) -> CaptionSelection:
    if not candidates:
        raise ValueError("at least one caption candidate is required")
    by_label = {candidate.label.upper(): candidate for candidate in candidates}
    if len(by_label) != len(candidates):
        raise ValueError("caption candidate labels must be unique")

    valid_scores = {
        label: score
        for label, score in scores.items()
        if label in by_label and score is not None
    }
    if valid_scores:
        highest = max(valid_scores.values())
        tied = [
            candidate.label.upper()
            for candidate in candidates
            if valid_scores.get(candidate.label.upper()) == highest
        ]
        choice = judge_choice.upper() if judge_choice else None
        selected_label = choice if choice in tied else tied[0]
    else:
        choice = judge_choice.upper() if judge_choice else None
        selected_label = choice if choice in by_label else candidates[0].label.upper()

    selected = by_label[selected_label]
    candidate_scores = [
        CandidateScore(label=candidate.label.upper(), score=scores.get(candidate.label.upper()))
        for candidate in candidates
    ]
    return CaptionSelection(
        selected_label=selected_label,
        selected_candidate_id=selected.candidate_id,
        judge_choice=judge_choice,
        scores=candidate_scores,
    )

