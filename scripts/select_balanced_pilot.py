"""Select a deterministic change/no-change pilot from an RS-Agent JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.per_class < 1:
        raise ValueError("--per-class must be positive")
    references: Dict[str, int] = {
        record["item_id"]: int(record["change_flag"])
        for record in read_jsonl(args.references)
    }
    rows = read_jsonl(args.input)
    selected = []
    counts = {0: 0, 1: 0}
    for flag in (0, 1):
        for row in rows:
            if references.get(row["item_id"]) != flag:
                continue
            selected.append(row)
            counts[flag] += 1
            if counts[flag] == args.per_class:
                break
        if counts[flag] < args.per_class:
            raise ValueError(
                "input contains only {} samples with change_flag={}".format(
                    counts[flag], flag
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": len(selected),
                "change_flag_counts": counts,
                "item_ids": [row["item_id"] for row in selected],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
