import json
from pathlib import Path

from rs_agent.domains.remote_sensing.dataset import iter_rs_cc_jsonl


def test_jsonl_accepts_normalized_and_legacy_records(tmp_path: Path) -> None:
    path = tmp_path / "captions.jsonl"
    records = [
        {"item_id": "test_1", "original_caption": "A building appeared.", "split": "test"},
        {"Original Caption": "A road disappeared."},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    parsed = list(iter_rs_cc_jsonl(path))
    assert parsed[0].item_id == "test_1"
    assert parsed[0].metadata == {"split": "test"}
    assert parsed[1].item_id == "item_000002"
    assert parsed[1].original_caption == "A road disappeared."


def test_jsonl_limit_counts_non_empty_records(tmp_path: Path) -> None:
    path = tmp_path / "captions.jsonl"
    path.write_text(
        '\n{"Original Caption":"first"}\n{"Original Caption":"second"}\n',
        encoding="utf-8",
    )
    assert len(list(iter_rs_cc_jsonl(path, limit=1))) == 1
