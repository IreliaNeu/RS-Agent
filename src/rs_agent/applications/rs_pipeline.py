"""CLI for the complete RS-CC to Knowledge Bridge to RS-VQA workflow."""

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
from rs_agent.domains.remote_sensing.config import load_rs_cc_config
from rs_agent.domains.remote_sensing.vqa_config import load_rs_vqa_config
from rs_agent.orchestration.agent_pipeline import RSAgentPipeline, RSAgentRequest
from rs_agent.providers.registry import ProviderRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete RS-Agent evidence pipeline.")
    parser.add_argument("--cc-config", type=Path, default=Path("configs/rs_cc.smoke.yaml"))
    parser.add_argument("--vqa-config", type=Path, default=Path("configs/rs_vqa.smoke.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--caption", required=True, help="Original Change-Agent pair caption")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--item-id", default="single_pair")
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_run_id() -> str:
    return "rs-agent-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


async def run(args: argparse.Namespace) -> dict:
    cc_config = load_rs_cc_config(args.cc_config)
    vqa_config = load_rs_vqa_config(args.vqa_config)
    cc_registry = ProviderRegistry(cc_config.model_registry())
    vqa_registry = ProviderRegistry(vqa_config.provider_registry())
    try:
        pipeline = RSAgentPipeline(
            cc_config,
            cc_registry,
            vqa_config,
            vqa_registry,
            JsonArtifactStore(args.artifact_dir),
        )
        result = await pipeline.run(
            RSAgentRequest(
                item_id=args.item_id,
                original_caption=args.caption,
                images=ImagePair(before=args.image_a, after=args.image_b),
                user_questions=args.question,
            ),
            args.run_id or default_run_id(),
        )
        return result.model_dump(mode="json")
    finally:
        await cc_registry.close()
        await vqa_registry.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    try:
        cc_config = load_rs_cc_config(args.cc_config)
        vqa_config = load_rs_vqa_config(args.vqa_config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "stages": ["rs_cc", "knowledge_bridge", "rs_vqa"],
                        "cc_profile": cc_config.profile,
                        "vqa_profile": vqa_config.profile,
                        "knowledge_payload": "selected_c_star_only",
                        "visual_input": "original_before_after_images_only",
                        "question_mode": "user" if args.question else "preset",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print("RS-Agent pipeline error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
