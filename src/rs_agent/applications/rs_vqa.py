"""Command-line entry point for paper-aligned RS-VQA on one image pair."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ImagePair
from rs_agent.domains.remote_sensing.vqa_agent import RSVQARequest
from rs_agent.domains.remote_sensing.vqa_config import load_rs_vqa_config
from rs_agent.orchestration.vqa_pipeline import RSVQAPipeline
from rs_agent.providers.registry import ProviderRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer preset or user RS-VQA questions with five visual models."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rs_vqa.paper.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--image-a", type=Path, required=True, help="Original before image")
    parser.add_argument("--image-b", type=Path, required=True, help="Original after image")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--item-id", default="single_pair")
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_run_id() -> str:
    return "rs-vqa-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


async def run(args: argparse.Namespace) -> dict:
    config = load_rs_vqa_config(args.config)
    registry = ProviderRegistry(config.provider_registry())
    try:
        pipeline = RSVQAPipeline(config, registry, JsonArtifactStore(args.artifact_dir))
        request = RSVQARequest(
            item_id=args.item_id,
            images=ImagePair(before=args.image_a, after=args.image_b),
            user_questions=args.question,
        )
        result = await pipeline.run(request, args.run_id or default_run_id())
        return result.model_dump(mode="json")
    finally:
        await registry.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    try:
        config = load_rs_vqa_config(args.config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "profile": config.profile,
                        "input_mode": "original_before_after_images_only",
                        "questions": args.question or config.default_template_ids,
                        "answer_models": [model.model for model in config.answer_models],
                        "selector": config.selector.model,
                        "evaluator": config.evaluator.model,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print("RS-VQA error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
