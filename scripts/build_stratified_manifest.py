"""Build a deterministic LEVIR-MCI subset stratified by ground-truth label values."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


def label_profile(path: Path) -> tuple[str, dict[str, int]]:
    values = Image.open(path).convert("L").getdata()
    counts = {"0": 0, "128": 0, "255": 0, "other": 0}
    for value in values:
        key = str(value) if value in {0, 128, 255} else "other"
        counts[key] += 1
    present = {key for key in ("128", "255", "other") if counts[key]}
    if not present:
        change_type = "no_change"
    elif present == {"128"}:
        change_type = "class_128_only"
    elif present == {"255"}:
        change_type = "class_255_only"
    else:
        change_type = "mixed_change_labels"
    return change_type, counts


def build_subset(
    input_path: Path,
    label_dir: Path,
    output_path: Path,
    count: int,
    seed: int,
) -> dict:
    with input_path.open("r", encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    groups = defaultdict(list)
    for row in source:
        label_path = label_dir / "{}.png".format(row["item_id"])
        if not label_path.is_file():
            raise ValueError("missing label mask: {}".format(label_path))
        change_type, counts = label_profile(label_path)
        enriched = {
            **row,
            "dataset": "LEVIR-MCI",
            "change_type": change_type,
            "ground_truth_label": str(label_path.resolve()),
            "ground_truth_label_pixels": counts,
        }
        groups[change_type].append(enriched)
    if count < 1 or count > len(source):
        raise ValueError("count must be between 1 and the input size")
    available_counts = {name: len(rows) for name, rows in groups.items()}
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    strata = sorted(groups)
    quota = count // len(strata)
    selected = []
    for name in strata:
        selected.extend(groups[name][:quota])
        groups[name] = groups[name][quota:]
    remaining = [row for name in strata for row in groups[name]]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    selected.sort(key=lambda row: row["item_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError("output already exists: {}".format(output_path))
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected_counts = defaultdict(int)
    for row in selected:
        selected_counts[row["change_type"]] += 1
    return {
        "input_count": len(source),
        "selected_count": len(selected),
        "seed": seed,
        "available_by_stratum": dict(sorted(available_counts.items())),
        "selected_by_stratum": dict(sorted(selected_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build_subset(args.input, args.label_dir, args.output, args.count, args.seed),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
