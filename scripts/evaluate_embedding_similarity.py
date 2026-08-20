"""Evaluate caption embeddings through an OpenAI-compatible embeddings API."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from rs_agent.evaluation.embedding_similarity import (
    caption_embedding_rows,
    summarize_embedding_rows,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def embeddings_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return base + "/embeddings" if base.endswith("/v1") else base + "/v1/embeddings"


def request_embeddings(
    texts: list[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int,
) -> dict[str, list[float]]:
    output = {}
    with httpx.Client(timeout=90.0) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.post(
                embeddings_url(base_url),
                headers={
                    "Authorization": "Bearer {}".format(api_key),
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": batch},
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda row: row["index"])
            if len(data) != len(batch):
                raise ValueError("embedding API returned a different batch size")
            output.update(
                {text: list(row["embedding"]) for text, row in zip(batch, data)}
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory is not empty: {}".format(args.output_dir))
    load_dotenv(args.env_file, override=False)
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError("missing API key environment variable: {}".format(args.api_key_env))
    items = {
        row["item_id"]: str(row.get("selected_caption") or "").strip()
        for row in read_jsonl(args.items)
    }
    references = {
        row["item_id"]: [str(text).strip() for text in row["references"]]
        for row in read_jsonl(args.references)
    }
    texts = sorted(
        {
            text
            for item_id, caption in items.items()
            for text in [caption, *references.get(item_id, [])]
            if text
        }
    )
    embeddings = request_embeddings(
        texts,
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        batch_size=args.batch_size,
    )
    rows = caption_embedding_rows(items, references, embeddings)
    summary = {
        **summarize_embedding_rows(rows),
        "model": args.model,
        "base_url": args.base_url,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "embedding_similarity.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["item_id"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
