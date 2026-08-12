"""Build a normalized multi-reference caption manifest from LEVIR-MCI annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rs_agent.evaluation.reference_metrics import (
    references_from_levir_annotations,
    write_reference_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = references_from_levir_annotations(args.annotations, args.split)
    write_reference_manifest(records, args.output)
    print(
        json.dumps(
            {"split": args.split, "record_count": len(records), "output": str(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
