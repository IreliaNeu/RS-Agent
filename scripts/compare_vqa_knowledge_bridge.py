"""Compare paired VQA exports with and without the Knowledge Bridge."""

import argparse
import json
from pathlib import Path

from rs_agent.evaluation.comparisons import (
    compare_knowledge_bridge_exports,
    write_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-export", type=Path, required=True)
    parser.add_argument("--with-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = compare_knowledge_bridge_exports(
        args.without_export,
        args.with_export,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    write_summary(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
