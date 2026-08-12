import csv
import json
from pathlib import Path

from test_batch_export import build_state

from rs_agent.evaluation.batch_export import export_batch


def test_export_adds_reference_metrics_and_request_telemetry(tmp_path: Path) -> None:
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
    summary = export_batch(state, output, references)

    assert summary["caption_reference_matched_items"] == 1
    assert summary["caption_reference_metrics"]["bleu_1"] == 1.0
    assert summary["caption_references_sha256"]
    assert summary["caption_reference_metrics_version"] == "levir_mci_caption_metrics_v1.1"
    assert len(summary["export_source_sha256"]) == 64
    with (output / "request_telemetry.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["role"] for row in rows} == {"generator", "selector", "evaluator"}
    with (output / "caption_reference_metrics.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        metric = next(csv.DictReader(handle))
    assert metric["change_flag_match"] == "True"
