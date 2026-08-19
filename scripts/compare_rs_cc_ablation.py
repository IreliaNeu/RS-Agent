"""Compare two RS-CC exports with paired bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rs_agent.evaluation.ablation import compare_exports
from rs_agent.evaluation.bootstrap import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-export", type=Path, required=True)
    parser.add_argument("--enhancement-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="text_only")
    parser.add_argument("--enhancement-name", default="mixed_image_text")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = compare_exports(
        args.baseline_export,
        args.enhancement_export,
        args.output_dir,
        baseline_name=args.baseline_name,
        enhancement_name=args.enhancement_name,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
