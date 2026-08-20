"""CLI for credential-safe provider and model preflight."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

from rs_agent.domains.remote_sensing.config import load_rs_cc_config
from rs_agent.domains.remote_sensing.vqa_config import load_rs_vqa_config
from rs_agent.providers.preflight import collect_provider_models, run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc-config", type=Path, default=Path("configs/rs_cc.smoke.yaml"))
    parser.add_argument("--vqa-config", type=Path, default=Path("configs/rs_vqa.smoke.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--network", action="store_true", help="Query each /models endpoint")
    parser.add_argument(
        "--probe-completions",
        action="store_true",
        help="Send one minimal chat completion to every configured model",
    )
    return parser


async def inspect(args: argparse.Namespace) -> dict:
    cc = load_rs_cc_config(args.cc_config)
    vqa = load_rs_vqa_config(args.vqa_config)
    provider_models = collect_provider_models(
        [
            (
                cc.providers,
                [*cc.caption_generators, cc.selector, cc.evaluator],
            ),
            (
                vqa.providers,
                [*vqa.answer_models, vqa.selector, vqa.evaluator],
            ),
        ]
    )
    image_model_ids = {
        model.model
        for model in cc.caption_generators
        if model.input_mode.value == "image_text"
    }
    image_model_ids.update(
        model.model for model in [*vqa.answer_models, vqa.selector, vqa.evaluator]
    )
    report = await run_preflight(
        provider_models,
        check_network=args.network or args.probe_completions,
        probe_completions=args.probe_completions,
        image_model_ids=image_model_ids,
    )
    return {
        "cc_profile": cc.profile,
        "vqa_profile": vqa.profile,
        **report.model_dump(mode="json"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    try:
        report = asyncio.run(inspect(args))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print("RS-Agent preflight error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
