"""Command-line entry point for text-only or capability-aware RS-CC experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from rs_agent.core.artifacts import JsonArtifactStore, export_jsonl
from rs_agent.core.schemas import ImagePair
from rs_agent.domains.remote_sensing.caption_agent import RSCCRequest
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig, load_rs_cc_config
from rs_agent.domains.remote_sensing.dataset import (
    RSCCInputRecord,
    iter_rs_cc_jsonl,
    materialize_records,
)
from rs_agent.orchestration.caption_pipeline import RSCCPipeline
from rs_agent.providers.registry import ProviderRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate five RS-CC candidates."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rs_cc.paper.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Change-Agent caption JSONL")
    source.add_argument("--caption", help="Run one original caption directly")
    parser.add_argument("--image-a", type=Path)
    parser.add_argument("--image-b", type=Path)
    parser.add_argument("--item-id", default="single_item")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_inputs(args: argparse.Namespace) -> List[RSCCInputRecord]:
    if args.caption is not None:
        return [
            RSCCInputRecord(
                item_id=args.item_id,
                original_caption=args.caption,
                image_a=args.image_a,
                image_b=args.image_b,
                metadata={},
            )
        ]
    return materialize_records(iter_rs_cc_jsonl(args.input, limit=args.limit))


def validate_images(config: RSCCExperimentConfig, records: Sequence[RSCCInputRecord]) -> None:
    if not config.requires_images:
        return
    missing = [
        record.item_id
        for record in records
        if record.image_a is None or record.image_b is None
    ]
    if missing:
        raise ValueError(
            "image-text RS-CC requires image_a/image_b for every item; missing: {}".format(
                missing[:10]
            )
        )


def dry_run_summary(
    config: RSCCExperimentConfig, records: Sequence[RSCCInputRecord]
) -> Dict[str, object]:
    return {
        "status": "valid",
        "profile": config.profile,
        "protocol": config.protocol.model_dump(mode="json"),
        "prompt_profile": config.prompt_profile.value,
        "input_mode": (
            "capability_aware_original_pair_and_caption"
            if config.requires_images
            else "text_only_original_caption"
        ),
        "record_count": len(records),
        "minimum_successful_candidates": config.minimum_successful_candidates,
        "caption_generators": [
            {
                "label": chr(ord("A") + index),
                "name": model.name,
                "model": model.model,
                "input_mode": model.input_mode.value,
            }
            for index, model in enumerate(config.caption_generators)
        ],
        "selector": config.selector.model,
        "evaluator": config.evaluator.model,
    }


async def run_batch(
    config: RSCCExperimentConfig,
    records: Sequence[RSCCInputRecord],
    artifact_dir: Path,
    run_id: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    registry = ProviderRegistry(config.model_registry())
    pipeline = RSCCPipeline(config, registry, JsonArtifactStore(artifact_dir))
    results: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    try:
        for record in records:
            images = (
                ImagePair(before=record.image_a, after=record.image_b)
                if record.image_a is not None and record.image_b is not None
                else None
            )
            request = RSCCRequest(
                item_id=record.item_id,
                original_caption=record.original_caption,
                images=images,
            )
            try:
                result = await pipeline.run(request, run_id=run_id)
            except Exception as exc:
                failures.append(
                    {
                        "item_id": record.item_id,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
                continue
            payload = result.model_dump(mode="json")
            payload["original_caption"] = record.original_caption
            payload["metadata"] = record.metadata
            results.append(payload)
    finally:
        await registry.close()
    return results, failures


def default_run_id() -> str:
    return "rs-cc-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(".env"), override=False)
    try:
        config = load_rs_cc_config(args.config)
        records = load_inputs(args)
        validate_images(config, records)
    except (OSError, ValueError) as exc:
        print("configuration/input error: {}".format(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(dry_run_summary(config, records), ensure_ascii=False, indent=2))
        return 0

    run_id = args.run_id or default_run_id()
    results, failures = asyncio.run(
        run_batch(config, records, args.artifact_dir, run_id)
    )
    output = args.output or args.artifact_dir / run_id / "rs_cc_results.jsonl"
    export_jsonl(output, [*results, *failures])
    print(
        json.dumps(
            {
                "run_id": run_id,
                "succeeded": len(results),
                "failed": len(failures),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
