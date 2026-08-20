from rs_agent.evaluation.paper_benchmark import build_paper_benchmark_summary


def row(item_id: str, model: str, score: int, change_type: str) -> dict:
    return {
        "item_id": item_id,
        "model_name": model,
        "provider": "provider",
        "model_id": model.lower(),
        "dataset": "LEVIR-MCI",
        "change_type": change_type,
        "success": True,
        "score": score,
        "selected": model == "A",
        "judge_choice": False,
    }


def test_paper_summary_groups_overall_and_change_type() -> None:
    rows = [
        row("one", "A", 10, "no_change"),
        row("two", "A", 8, "building"),
        row("one", "B", 6, "no_change"),
        row("two", "B", 4, "building"),
    ]

    summary = build_paper_benchmark_summary(rows, [])
    overall_a = next(
        item
        for item in summary
        if item["scope"] == "overall" and item["model_name"] == "A"
    )
    building_a = next(
        item
        for item in summary
        if item["scope"] == "change_type"
        and item["model_name"] == "A"
        and item["change_type"] == "building"
    )

    assert overall_a["mean_score"] == 9
    assert overall_a["population_std"] == 1
    assert building_a["mean_score"] == 8
    assert building_a["candidate_count"] == 1
