"""CLI for resumable RS-Agent JSONL batch experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

from rs_agent.core.artifacts import JsonArtifactStore
from rs_agent.core.schemas import ImagePair, TaskType
from rs_agent.domains.remote_sensing.config import load_rs_cc_config
from rs_agent.domains.remote_sensing.mask_evidence import MaskEvidenceRequest, MaskSource
from rs_agent.domains.remote_sensing.vqa_config import load_rs_vqa_config
from rs_agent.experiments.batch import BatchRunner, load_batch_items
from rs_agent.experiments.cache import ResultCache
from rs_agent.experiments.identity import BatchItem, build_experiment_identity
from rs_agent.experiments.state import BatchStateStore
from rs_agent.orchestration.agent_pipeline import RSAgentPipeline, RSAgentRequest
from rs_agent.providers.registry import ProviderRegistry

TASK_TYPES = {
    "caption": TaskType.CAPTION_ENRICHMENT,
    "vqa": TaskType.VQA,
    "combined": TaskType.COMBINED,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--cc-config", type=Path, default=Path("configs/rs_cc.smoke.yaml"))
    parser.add_argument("--vqa-config", type=Path, default=Path("configs/rs_vqa.smoke.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--task-type", choices=tuple(TASK_TYPES), default="combined")
    parser.add_argument("--without-knowledge-bridge", action="store_true")
    parser.add_argument("--without-masks", action="store_true")
    parser.add_argument("--mask-source", choices=tuple(item.value for item in MaskSource), default="predicted")
    parser.add_argument("--mask-min-component-pixels", type=int, default=1)
    parser.add_argument("--mask-min-changed-ratio", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--state-dir", type=Path, default=Path("batch-state"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def options(args: argparse.Namespace) -> dict:
    return {
        "task_type": TASK_TYPES[args.task_type].value,
        "use_knowledge_bridge": not args.without_knowledge_bridge,
        "use_masks": not args.without_masks,
        "mask_source": args.mask_source,
        "mask_min_component_pixels": args.mask_min_component_pixels,
        "mask_min_changed_ratio": args.mask_min_changed_ratio,
    }


def mask_request(item: BatchItem, args: argparse.Namespace) -> Optional[MaskEvidenceRequest]:
    if args.without_masks or item.predicted_mask is None:
        return None
    return MaskEvidenceRequest(
        path=item.predicted_mask,
        source=MaskSource(args.mask_source),
        min_component_pixels=args.mask_min_component_pixels,
        min_changed_ratio=args.mask_min_changed_ratio,
    )


async def execute(args: argparse.Namespace) -> dict:
    items = load_batch_items(args.input)
    cc_config = load_rs_cc_config(args.cc_config)
    vqa_config = load_rs_vqa_config(args.vqa_config)
    experiment = build_experiment_identity(
        input_path=args.input,
        config_paths={"rs_cc": args.cc_config, "rs_vqa": args.vqa_config},
        options=options(args),
        items=items,
    )
    state_path = args.state_dir / args.batch_id / "state.json"
    if args.dry_run:
        return {
            "status": "valid",
            "batch_id": args.batch_id,
            "item_count": len(items),
            "experiment_fingerprint": experiment.fingerprint,
            "state_path": str(state_path),
            "options": experiment.options,
        }

    cc_registry = ProviderRegistry(cc_config.model_registry())
    vqa_registry = ProviderRegistry(vqa_config.provider_registry())
    pipeline = RSAgentPipeline(
        cc_config,
        cc_registry,
        vqa_config,
        vqa_registry,
        JsonArtifactStore(args.artifact_dir),
    )

    async def run_item(item: BatchItem, run_id: str) -> str:
        result = await pipeline.run(
            RSAgentRequest(
                item_id=item.item_id,
                original_caption=item.original_caption,
                images=ImagePair(before=item.image_a, after=item.image_b),
                task_type=TASK_TYPES[args.task_type],
                user_questions=item.user_questions,
                use_knowledge_bridge=not args.without_knowledge_bridge,
                mask=mask_request(item, args),
            ),
            run_id,
        )
        return result.result_artifact

    try:
        runner = BatchRunner(
            batch_id=args.batch_id,
            items=items,
            experiment=experiment,
            state_store=BatchStateStore(state_path),
            run_item=run_item,
            concurrency=args.concurrency,
            retry_failed=args.retry_failed,
            max_items=args.max_items,
            result_cache=ResultCache(args.cache_dir) if args.cache_dir else None,
        )
        state = await runner.run()
        return {
            "batch_id": state.batch_id,
            "status": state.status.value,
            "counts": state.counts(),
            "experiment_fingerprint": state.experiment.fingerprint,
            "state_path": str(state_path),
        }
    finally:
        await cc_registry.close()
        await vqa_registry.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    try:
        result = asyncio.run(execute(args))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("counts", {}).get("failed", 0) == 0 else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print("RS-Agent batch error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
