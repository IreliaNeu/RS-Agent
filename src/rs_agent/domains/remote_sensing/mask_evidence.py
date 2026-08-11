"""Structured evidence extraction for optional change masks."""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from pydantic import Field

from rs_agent.core.schemas import StrictModel


class MaskSource(str, Enum):
    PREDICTED = "predicted"
    GROUND_TRUTH = "ground_truth"
    EXTERNAL = "external"


class MaskEvidenceRequest(StrictModel):
    path: Path
    source: MaskSource = MaskSource.PREDICTED
    class_labels: Dict[int, str] = Field(
        default_factory=lambda: {
            0: "background",
            1: "road_change",
            2: "building_change",
            255: "change",
        }
    )
    min_component_pixels: int = Field(default=1, ge=1)
    min_changed_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class MaskClassStats(StrictModel):
    value: int
    label: str
    pixel_count: int = Field(ge=0)
    ratio: float = Field(ge=0.0, le=1.0)


class MaskComponent(StrictModel):
    pixel_count: int = Field(ge=1)
    ratio: float = Field(gt=0.0, le=1.0)
    bounding_box_xyxy: List[int] = Field(min_length=4, max_length=4)


class MaskEvidenceSummary(StrictModel):
    path: str
    source: MaskSource
    sha256: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    class_stats: List[MaskClassStats]
    changed_pixel_count: int = Field(ge=0)
    changed_ratio: float = Field(ge=0.0, le=1.0)
    component_count: int = Field(ge=0)
    significant_component_count: int = Field(ge=0)
    largest_component: Optional[MaskComponent] = None
    has_change: bool
    interpretation: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _components(
    pixels: List[int], width: int, height: int
) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    """Return 4-connected non-background component sizes and bounding boxes."""

    visited = bytearray(width * height)
    found: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for start, value in enumerate(pixels):
        if value == 0 or visited[start]:
            continue
        queue = deque([start])
        visited[start] = 1
        size = 0
        min_x = max_x = start % width
        min_y = max_y = start // width
        while queue:
            index = queue.popleft()
            x = index % width
            y = index // width
            size += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            if x > 0:
                neighbor = index - 1
                if pixels[neighbor] != 0 and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
            if x + 1 < width:
                neighbor = index + 1
                if pixels[neighbor] != 0 and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
            if y > 0:
                neighbor = index - width
                if pixels[neighbor] != 0 and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
            if y + 1 < height:
                neighbor = index + width
                if pixels[neighbor] != 0 and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
        found.append((size, (min_x, min_y, max_x, max_y)))
    return found


def analyze_mask(request: MaskEvidenceRequest) -> MaskEvidenceSummary:
    path = request.path.resolve()
    if not path.is_file():
        raise ValueError("mask file does not exist: {}".format(path))

    with Image.open(path) as image:
        if image.mode not in {"1", "L", "P"}:
            raise ValueError(
                "mask must be a single-channel class-index image, got mode {}".format(
                    image.mode
                )
            )
        width, height = image.size
        pixels = [int(value) for value in image.getdata()]

    total = width * height
    counts = Counter(pixels)
    class_stats = [
        MaskClassStats(
            value=value,
            label=request.class_labels.get(value, "class_{}".format(value)),
            pixel_count=count,
            ratio=count / total,
        )
        for value, count in sorted(counts.items())
    ]
    changed_pixels = total - counts.get(0, 0)
    changed_ratio = changed_pixels / total
    all_components = _components(pixels, width, height)
    significant = [
        component
        for component in all_components
        if component[0] >= request.min_component_pixels
    ]
    largest = max(significant, key=lambda component: component[0], default=None)
    largest_summary = None
    if largest is not None:
        largest_summary = MaskComponent(
            pixel_count=largest[0],
            ratio=largest[0] / total,
            bounding_box_xyxy=list(largest[1]),
        )
    has_change = bool(significant) and changed_ratio >= request.min_changed_ratio
    interpretation = (
        "The optional mask contains significant non-background change regions."
        if has_change
        else "The optional mask contains no significant non-background change region."
    )
    return MaskEvidenceSummary(
        path=str(path),
        source=request.source,
        sha256=_sha256(path),
        width=width,
        height=height,
        class_stats=class_stats,
        changed_pixel_count=changed_pixels,
        changed_ratio=changed_ratio,
        component_count=len(all_components),
        significant_component_count=len(significant),
        largest_component=largest_summary,
        has_change=has_change,
        interpretation=interpretation,
    )
