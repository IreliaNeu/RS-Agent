"""CLI for the paper-aligned RS-Agent workflow."""

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
from rs_agent.core.schemas import ImagePair, TaskType
from rs_agent.domains.remote_sensing.config import load_rs_cc_config
from rs_agent.domains.remote_sensing.mask_evidence import MaskEvidenceRequest, MaskSource
from rs_agent.domains.remote_sensing.vqa_config import load_rs_vqa_config
from rs_agent.orchestration.agent_pipeline import RSAgentPipeline, RSAgentRequest
from rs_agent.orchestration.main_agent import plan_request
from rs_agent.providers.registry import ProviderRegistry

TASK_TYPES = {
    "caption": TaskType.CAPTION_ENRICHMENT,
    "vqa": TaskType.VQA,
    "combined": TaskType.COMBINED,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RS-Agent evidence pipeline.")
    parser.add_argument("--cc-config", type=Path, default=Path("configs/rs_cc.smoke.yaml"))
    parser.add_argument("--vqa-config", type=Path, default=Path("configs/rs_vqa.smoke.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--caption", required=True, help="Original Change-Agent pair caption")
    parser.add_argument("--task-type", choices=tuple(TASK_TYPES), default="combined")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument(
        "--without-knowledge-bridge",
        action="store_true",
        help="Run the paper's RS-VQA knowledge-bridge ablation path",
    )
    parser.add_argument("--mask", type=Path, help="Optional class-index change mask")
    parser.add_argument(
        "--mask-source",
        choices=tuple(source.value for source in MaskSource),
        default=MaskSource.PREDICTED.value,
    )
    parser.add_argument("--mask-min-component-pixels", type=int, default=1)
    parser.add_argument("--mask-min-changed-ratio", type=float, default=0.0)
    parser.add_argument("--item-id", default="single_pair")
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_run_id() -> str:
    return "rs-agent-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def mask_request(args: argparse.Namespace) -> Optional[MaskEvidenceRequest]:
    if args.mask is None:
        return None
    return MaskEvidenceRequest(
        path=args.mask,
        source=MaskSource(args.mask_source),
        min_component_pixels=args.mask_min_component_pixels,
        min_changed_ratio=args.mask_min_changed_ratio,
    )


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
                task_type=TASK_TYPES[args.task_type],
                user_questions=args.question,
                use_knowledge_bridge=not args.without_knowledge_bridge,
                mask=mask_request(args),
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
        task_type = TASK_TYPES[args.task_type]
        plan = plan_request(
            task_type,
            use_knowledge_bridge=not args.without_knowledge_bridge,
            has_mask=args.mask is not None,
        )
        cc_config = load_rs_cc_config(args.cc_config)
        vqa_config = load_rs_vqa_config(args.vqa_config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "task_type": task_type.value,
                        "stages": [stage.value for stage in plan.stages],
                        "cc_profile": cc_config.profile,
                        "vqa_profile": vqa_config.profile,
                        "knowledge_payload": (
                            "selected_c_star_only"
                            if plan.use_knowledge_bridge
                            else "disabled"
                        ),
                        "visual_input": "original_before_after_images_only",
                        "question_mode": "user" if args.question else "preset",
                        "mask_input_to_vlm": False,
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
