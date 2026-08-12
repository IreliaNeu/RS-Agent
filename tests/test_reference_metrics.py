import json
from pathlib import Path

import pytest

from rs_agent.evaluation.reference_metrics import (
    CaptionReference,
    caption_change_flag,
    evaluate_caption,
    load_reference_manifest,
    references_from_levir_annotations,
    sentence_bleu,
)


def test_exact_reference_has_perfect_caption_metrics() -> None:
    text = "Two buildings were constructed beside the road"
    record = evaluate_caption(
        "x", text, CaptionReference(item_id="x", split="test", references=[text], change_flag=1)
    )
    assert record.bleu_1 == pytest.approx(1.0)
    assert record.bleu_4 == pytest.approx(1.0)
    assert record.rouge_l == pytest.approx(1.0)
    assert record.change_flag_match is True


def test_no_change_classifier_and_short_bleu_definition() -> None:
    assert caption_change_flag("There is no difference between the two images.") == 0
    assert caption_change_flag("The scene remains unchanged from before.") == 0
    assert caption_change_flag("There are no discernible differences.") == 0
    assert caption_change_flag("A new road was constructed.") == 1
    assert sentence_bleu("no change", ["no change"], 4) == 0.0


def test_levir_annotation_normalization_and_duplicate_guard(tmp_path: Path) -> None:
    source = tmp_path / "captions.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "filename": "test_1.png",
                        "split": "test",
                        "changeflag": 0,
                        "sentences": [{"raw": "No change."}, {"raw": "The scenes seem identical."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records = references_from_levir_annotations(source, "test")
    assert records[0].item_id == "test_1"
    assert records[0].references == ["No change", "The scenes seem identical"]

    manifest = tmp_path / "references.jsonl"
    line = json.dumps(records[0].model_dump(mode="json"))
    manifest.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate item_id"):
        load_reference_manifest(manifest)
