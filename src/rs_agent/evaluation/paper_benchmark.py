"""Paper-style LLM-as-Judge summaries by model and change type."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def _group_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    stage: str,
) -> Dict[Tuple[str, ...], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = str(row.get("dataset") or "unspecified")
        model = (
            str(row["model_name"]),
            str(row["provider"]),
            str(row["model_id"]),
        )
        groups[(stage, "overall", dataset, "", "", *model)].append(row)
        change_type = str(row.get("change_type") or "unspecified")
        groups[(stage, "change_type", dataset, change_type, "", *model)].append(row)
        question_id = str(row.get("question_id") or "")
        if stage == "rs_vqa" and question_id:
            groups[(stage, "question", dataset, change_type, question_id, *model)].append(
                row
            )
    return groups


def build_paper_benchmark_summary(
    caption_rows: Iterable[Mapping[str, Any]],
    vqa_rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate Judge scores using the mean/std protocol reported in the paper."""
    groups = _group_rows(caption_rows, stage="rs_cc")
    for key, rows in _group_rows(vqa_rows, stage="rs_vqa").items():
        groups[key].extend(rows)

    output: List[Dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        stage, scope, dataset, change_type, question_id, name, provider, model_id = key
        scores = [float(row["score"]) for row in rows if row.get("score") is not None]
        successes = [row for row in rows if bool(row.get("success"))]
        output.append(
            {
                "stage": stage,
                "scope": scope,
                "dataset": dataset,
                "change_type": change_type,
                "question_id": question_id,
                "model_name": name,
                "provider": provider,
                "model_id": model_id,
                "candidate_count": len(rows),
                "successful_count": len(successes),
                "score_count": len(scores),
                "mean_score": statistics.fmean(scores) if scores else None,
                "population_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "selected_count": sum(bool(row.get("selected")) for row in rows),
                "judge_choice_count": sum(
                    bool(row.get("judge_choice")) for row in rows
                ),
            }
        )
    return output
