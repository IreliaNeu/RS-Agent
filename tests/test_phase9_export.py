import csv
import json
from pathlib import Path

from test_batch_export import build_state

from rs_agent.evaluation.batch_export import export_batch


def test_export_writes_candidate_metrics_and_bootstrap_intervals(tmp_path: Path) -> None:
    state = build_state(tmp_path)
    references = tmp_path / "references.jsonl"
    references.write_text(
        json.dumps(
            {
                "item_id": "item-1",
                "split": "test",
                "references": ["No change is visible."],
                "change_flag": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "export"

    summary = export_batch(
        state,
        output,
        references,
        bootstrap_samples=100,
        bootstrap_seed=19,
    )

    interval = summary["caption_reference_metric_intervals"]["bleu_1"]
    assert interval["mean"] == interval["lower"] == interval["upper"] == 1.0
    assert summary["bootstrap"]["samples"] == 100
    assert summary["bootstrap"]["seed"] == 19
    assert summary["caption_candidate_reference_rows"] == 2

    with (output / "caption_candidate_reference_metrics.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        candidates = {row["model_name"]: row for row in csv.DictReader(handle)}
    assert candidates["Model A"]["bleu_1"] == "1.0"
    assert candidates["Model B"]["bleu_1"] == ""

    with (output / "caption_candidate_summary.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    text_mode = next(
        row
        for row in summary_rows
        if row["scope"] == "input_mode" and row["input_mode"] == "text_only"
    )
    assert text_mode["candidate_count"] == "2"
    assert text_mode["metric_count"] == "1"


def test_export_rejects_invalid_bootstrap_configuration(tmp_path: Path) -> None:
    state = build_state(tmp_path)
    try:
        export_batch(state, tmp_path / "output", bootstrap_samples=0)
    except ValueError as exc:
        assert "bootstrap_samples" in str(exc)
    else:
        raise AssertionError("invalid bootstrap configuration was accepted")
