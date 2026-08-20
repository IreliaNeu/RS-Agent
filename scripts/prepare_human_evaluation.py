"""Prepare deterministic blind original-versus-agent caption trials."""

import argparse
import json
from pathlib import Path

from rs_agent.evaluation.human_eval import prepare_blind_trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_blind_trials(args.items, args.output_dir, args.seed),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
