import json
from pathlib import Path

from PIL import Image

from scripts.build_stratified_manifest import build_subset


def test_stratified_manifest_adds_ground_truth_metadata(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    rows = []
    values = [0, 0, 128, 128, 255, 255]
    for index, value in enumerate(values):
        item_id = "item-{}".format(index)
        Image.new("L", (2, 2), color=value).save(labels / "{}.png".format(item_id))
        rows.append(
            {
                "item_id": item_id,
                "original_caption": "Caption.",
                "image_a": "a.png",
                "image_b": "b.png",
            }
        )
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "subset.jsonl"

    summary = build_subset(source, labels, output, count=3, seed=42)
    selected = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["selected_count"] == 3
    assert set(summary["selected_by_stratum"]) == {
        "no_change",
        "class_128_only",
        "class_255_only",
    }
    assert all(row["dataset"] == "LEVIR-MCI" for row in selected)
