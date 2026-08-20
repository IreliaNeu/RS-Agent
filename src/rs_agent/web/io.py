"""Validated file and manifest inputs for the Streamlit application."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Optional

from PIL import Image
from pydantic import Field

from rs_agent.core.schemas import StrictModel

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 8192 * 8192


class DemoSample(StrictModel):
    item_id: str
    original_caption: str
    image_a: Path
    image_b: Path
    predicted_mask: Optional[Path] = None
    user_questions: List[str] = Field(default_factory=list)


def load_demo_manifest(path: Path) -> List[DemoSample]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("demo manifest does not exist: {}".format(resolved))
    samples = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                for field in ("image_a", "image_b", "predicted_mask"):
                    value = data.get(field)
                    if value and not Path(value).is_absolute():
                        data[field] = resolved.parent / value
                sample = DemoSample.model_validate(data)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    "invalid demo manifest line {}: {}".format(line_number, exc)
                ) from exc
            for field in ("image_a", "image_b", "predicted_mask"):
                value = getattr(sample, field)
                if value is not None and not value.is_file():
                    raise ValueError(
                        "demo sample {} {} does not exist: {}".format(
                            sample.item_id, field, value
                        )
                    )
            samples.append(sample)
    if not samples:
        raise ValueError("demo manifest is empty")
    if len({sample.item_id for sample in samples}) != len(samples):
        raise ValueError("demo manifest item IDs must be unique")
    return samples


def save_uploaded_image(
    data: bytes,
    filename: str,
    directory: Path,
    *,
    kind: Literal["rgb", "mask"],
) -> Path:
    if not data:
        raise ValueError("uploaded image is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("uploaded image exceeds 20 MB")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("uploaded image must be PNG or JPEG")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("uploaded image dimensions are too large")
            if kind == "rgb" and image.mode not in {"RGB", "RGBA", "L"}:
                raise ValueError(
                    "uploaded scene image has unsupported mode {}".format(image.mode)
                )
            if kind == "mask" and image.mode not in {"1", "L", "P"}:
                raise ValueError("uploaded mask must be a single-channel class image")
    except OSError as exc:
        raise ValueError("uploaded file is not a valid image") from exc
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "{}{}".format(digest, suffix)
    if not path.exists():
        path.write_bytes(data)
    return path
