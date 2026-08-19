from rs_agent.evaluation.candidate_metrics import (
    build_candidate_metric_rows,
    summarize_candidate_metrics,
)
from rs_agent.evaluation.reference_metrics import CaptionReference


def candidate(label, mode, model, text, success=True, selected=False):
    return {
        "item_id": "item-1",
        "label": label,
        "input_mode": mode,
        "model_name": model,
        "provider": "fake",
        "model_id": "vendor/{}".format(model),
        "success": success,
        "selected": selected,
        "score": 9 if success else None,
        "text": text,
    }


def test_candidate_metrics_preserve_failures_and_summarize_modes() -> None:
    references = {
        "item-1": CaptionReference(
            item_id="item-1",
            split="test",
            references=["No change is visible."],
            change_flag=0,
        )
    }
    rows = build_candidate_metric_rows(
        [
            candidate("A", "text_only", "a", "No change is visible.", selected=True),
            candidate("B", "image_text", "b", "", success=False),
        ],
        references,
    )
    assert rows[0]["bleu_1"] == 1.0
    assert rows[1]["bleu_1"] is None
    assert rows[1]["reference_matched"] is True

    summary = summarize_candidate_metrics(
        rows, bootstrap_samples=50, confidence=0.95, seed=3
    )
    by_mode = {
        row["input_mode"]: row
        for row in summary
        if row["scope"] == "input_mode"
    }
    assert by_mode["text_only"]["bleu_1_mean"] == 1.0
    assert by_mode["image_text"]["metric_count"] == 0
