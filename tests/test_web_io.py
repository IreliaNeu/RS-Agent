import json
from pathlib import Path

import pytest
from PIL import Image

from rs_agent.web.io import load_demo_manifest, save_uploaded_image


def image_bytes(tmp_path: Path, mode: str = "RGB") -> bytes:
    path = tmp_path / "source.png"
    Image.new(mode, (8, 8), color=0).save(path)
    return path.read_bytes()


def test_uploads_are_content_addressed_and_validated(tmp_path: Path) -> None:
    data = image_bytes(tmp_path)
    first = save_uploaded_image(data, "before.png", tmp_path / "uploads", kind="rgb")
    second = save_uploaded_image(data, "renamed.png", tmp_path / "uploads", kind="rgb")

    assert first == second
    assert first.is_file()


def test_rgb_image_cannot_be_used_as_class_mask(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single-channel"):
        save_uploaded_image(
            image_bytes(tmp_path),
            "mask.png",
            tmp_path / "uploads",
            kind="mask",
        )


def test_manifest_requires_existing_unique_sample_files(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (8, 8)).save(image_a)
    Image.new("RGB", (8, 8)).save(image_b)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "item_id": "sample-1",
                "original_caption": "No change.",
                "image_a": str(image_a),
                "image_b": str(image_b),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_demo_manifest(manifest)

    assert samples[0].item_id == "sample-1"
    assert samples[0].image_a == image_a
