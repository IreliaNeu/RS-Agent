"""Embedding-based human-human and human-agent caption similarity."""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Mapping, Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise ValueError("embedding vectors must be non-zero")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def caption_embedding_rows(
    agent_captions: Mapping[str, str],
    human_references: Mapping[str, Sequence[str]],
    embeddings: Mapping[str, Sequence[float]],
) -> List[Dict[str, object]]:
    """Build per-item similarity rows using all human references."""
    rows: List[Dict[str, object]] = []
    for item_id in sorted(agent_captions):
        references = list(human_references.get(item_id, []))
        if not references:
            continue
        agent = agent_captions[item_id]
        agent_scores = [
            cosine_similarity(embeddings[agent], embeddings[reference])
            for reference in references
        ]
        human_scores = [
            cosine_similarity(embeddings[left], embeddings[right])
            for index, left in enumerate(references)
            for right in references[index + 1 :]
        ]
        rows.append(
            {
                "item_id": item_id,
                "reference_count": len(references),
                "human_agent_similarity": statistics.fmean(agent_scores),
                "human_human_similarity": (
                    statistics.fmean(human_scores) if human_scores else None
                ),
            }
        )
    return rows


def summarize_embedding_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    agent = [float(row["human_agent_similarity"]) for row in rows]
    human = [
        float(row["human_human_similarity"])
        for row in rows
        if row.get("human_human_similarity") is not None
    ]
    return {
        "item_count": len(rows),
        "mean_human_agent_similarity": statistics.fmean(agent) if agent else None,
        "mean_human_human_similarity": statistics.fmean(human) if human else None,
    }
