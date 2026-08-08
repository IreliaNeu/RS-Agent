"""Encoding and provenance for the original bi-temporal image pair."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path

from rs_agent.core.schemas import ImagePair, StrictModel


class EncodedImagePair(StrictModel):
    before_data_url: str
    after_data_url: str
    before_sha256: str
    after_sha256: str


def _encode_image(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise ValueError("image does not exist: {}".format(path))
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("unsupported image format: {}".format(path))
    encoded = base64.b64encode(data).decode("ascii")
    return "data:{};base64,{}".format(mime_type, encoded), hashlib.sha256(data).hexdigest()


def encode_original_image_pair(images: ImagePair) -> EncodedImagePair:
    before_url, before_hash = _encode_image(images.before)
    after_url, after_hash = _encode_image(images.after)
    return EncodedImagePair(
        before_data_url=before_url,
        after_data_url=after_url,
        before_sha256=before_hash,
        after_sha256=after_hash,
    )
