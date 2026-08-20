"""Compare two Judge-specific RS-Agent exports."""

import argparse
import json
from pathlib import Path

from rs_agent.evaluation.comparisons import compare_judge_exports, write_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-export", type=Path, required=True)
    parser.add_argument("--right-export", type=Path, required=True)
    parser.add_argument("--stage", choices=["rs_cc", "rs_vqa"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compare_judge_exports(args.left_export, args.right_export, args.stage)
    write_summary(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
