from pathlib import Path

from PIL import Image

from rs_agent.domains.remote_sensing.mask_evidence import (
    MaskEvidenceRequest,
    MaskSource,
    analyze_mask,
)


def test_all_background_mask_is_no_change(tmp_path: Path) -> None:
    path = tmp_path / "no-change.png"
    Image.new("L", (4, 3), color=0).save(path)

    summary = analyze_mask(MaskEvidenceRequest(path=path))

    assert summary.source == MaskSource.PREDICTED
    assert summary.changed_pixel_count == 0
    assert summary.changed_ratio == 0.0
    assert summary.component_count == 0
    assert summary.largest_component is None
    assert summary.has_change is False


def test_multiclass_mask_reports_classes_components_and_bbox(tmp_path: Path) -> None:
    path = tmp_path / "changed.png"
    image = Image.new("L", (5, 4), color=0)
    image.putdata(
        [
            0, 1, 1, 0, 2,
            0, 1, 0, 0, 2,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ]
    )
    image.save(path)

    summary = analyze_mask(
        MaskEvidenceRequest(path=path, min_component_pixels=3)
    )

    stats = {item.value: item for item in summary.class_stats}
    assert stats[1].label == "road_change"
    assert stats[1].pixel_count == 3
    assert stats[2].label == "building_change"
    assert stats[2].pixel_count == 2
    assert summary.changed_pixel_count == 5
    assert summary.changed_ratio == 0.25
    assert summary.component_count == 2
    assert summary.significant_component_count == 1
    assert summary.largest_component is not None
    assert summary.largest_component.pixel_count == 3
    assert summary.largest_component.bounding_box_xyxy == [1, 0, 2, 1]
    assert summary.has_change is True


def test_rgb_mask_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(path)

    try:
        analyze_mask(MaskEvidenceRequest(path=path))
    except ValueError as exc:
        assert "single-channel" in str(exc)
    else:
        raise AssertionError("RGB masks must not be interpreted as class indices")
