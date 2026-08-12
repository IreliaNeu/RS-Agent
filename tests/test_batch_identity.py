from pathlib import Path

from PIL import Image

from rs_agent.experiments.identity import BatchItem, build_experiment_identity


def make_item(tmp_path: Path, item_id: str = "pair-1") -> BatchItem:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(before)
    Image.new("RGB", (2, 2), color=(3, 2, 1)).save(after)
    return BatchItem(
        item_id=item_id,
        original_caption="The scene is unchanged.",
        image_a=before,
        image_b=after,
    )


def test_experiment_identity_changes_with_options(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("profile: test\n", encoding="utf-8")
    item = make_item(tmp_path)

    first = build_experiment_identity(
        input_path=input_path,
        config_paths={"cc": config},
        options={"task": "caption"},
        items=[item],
    )
    second = build_experiment_identity(
        input_path=input_path,
        config_paths={"cc": config},
        options={"task": "combined"},
        items=[item],
    )

    assert first.fingerprint != second.fingerprint
    assert len(first.item_fingerprints[item.item_id]) == 64


def test_item_fingerprint_uses_file_content_not_absolute_path(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    first = make_item(one)
    second = make_item(two)

    assert first.fingerprint() == second.fingerprint()
