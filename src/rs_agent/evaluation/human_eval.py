"""Deterministic blind human-evaluation preparation and scoring."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL records must be objects")
                rows.append(value)
    return rows


def prepare_blind_trials(items_path: Path, output_dir: Path, seed: int = 42) -> dict:
    """Blind original versus selected captions while storing the key separately."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(items_path)
    item_ids = [str(row.get("item_id") or "") for row in rows]
    if not rows or any(not item_id for item_id in item_ids):
        raise ValueError("items must contain non-empty item_id values")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item_id values must be unique")

    rng = random.Random(seed)
    trials = []
    keys = []
    for index, row in enumerate(sorted(rows, key=lambda value: value["item_id"]), 1):
        original = str(row.get("original_caption") or "").strip()
        agent = str(row.get("selected_caption") or "").strip()
        if not original or not agent:
            raise ValueError("item {} lacks one comparison caption".format(row["item_id"]))
        agent_side = "A" if rng.getrandbits(1) == 0 else "B"
        trial_id = "T{:04d}".format(index)
        text_a, text_b = (agent, original) if agent_side == "A" else (original, agent)
        trials.append(
            {
                "trial_id": trial_id,
                "item_id": row["item_id"],
                "text_a": text_a,
                "text_b": text_b,
            }
        )
        keys.append(
            {
                "trial_id": trial_id,
                "item_id": row["item_id"],
                "agent_side": agent_side,
                "original_side": "B" if agent_side == "A" else "A",
            }
        )

    with (output_dir / "blind_trials.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["trial_id", "item_id", "text_a", "text_b"]
        )
        writer.writeheader()
        writer.writerows(trials)
    with (output_dir / "answer_key.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for row in keys:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    protocol = {
        "seed": seed,
        "trial_count": len(trials),
        "response_columns": ["trial_id", "rater_id", "preferred_side"],
        "preferred_side_values": ["A", "B", "tie"],
        "blinding": "answer_key.jsonl must not be distributed to raters",
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return protocol


def score_blind_responses(key_path: Path, responses_path: Path) -> dict:
    """Score blind A/B/tie choices against the separated answer key."""
    key = {row["trial_id"]: row for row in _read_jsonl(key_path)}
    responses = []
    with responses_path.open("r", encoding="utf-8", newline="") as handle:
        responses.extend(csv.DictReader(handle))
    seen = set()
    overall: Counter = Counter()
    by_rater: Dict[str, Counter] = {}
    normalized_by_trial: Dict[str, List[str]] = {}
    for row in responses:
        trial_id = str(row.get("trial_id") or "")
        rater_id = str(row.get("rater_id") or "").strip()
        preferred = str(row.get("preferred_side") or "").strip()
        if trial_id not in key or not rater_id or preferred not in {"A", "B", "tie"}:
            raise ValueError("invalid human-evaluation response: {}".format(row))
        if (trial_id, rater_id) in seen:
            raise ValueError("duplicate response for {} and {}".format(trial_id, rater_id))
        seen.add((trial_id, rater_id))
        outcome = (
            "tie"
            if preferred == "tie"
            else "agent"
            if preferred == key[trial_id]["agent_side"]
            else "original"
        )
        overall[outcome] += 1
        by_rater.setdefault(rater_id, Counter())[outcome] += 1
        normalized_by_trial.setdefault(trial_id, []).append(outcome)

    pair_total = 0
    pair_agreements = 0
    for values in normalized_by_trial.values():
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                pair_total += 1
                pair_agreements += int(values[left] == values[right])
    count = sum(overall.values())
    return {
        "response_count": count,
        "rater_count": len(by_rater),
        "agent_wins": overall["agent"],
        "original_wins": overall["original"],
        "ties": overall["tie"],
        "agent_preference_rate_excluding_ties": (
            overall["agent"] / (overall["agent"] + overall["original"])
            if overall["agent"] + overall["original"]
            else None
        ),
        "pairwise_inter_rater_agreement": (
            pair_agreements / pair_total if pair_total else None
        ),
        "by_rater": {name: dict(counts) for name, counts in sorted(by_rater.items())},
    }
