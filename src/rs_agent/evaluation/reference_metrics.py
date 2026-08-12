"""Dependency-light LEVIR-MCI caption metrics with explicit definitions."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from pydantic import Field

from rs_agent.core.schemas import StrictModel

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
REFERENCE_METRICS_VERSION = "levir_mci_caption_metrics_v1.1"

NO_CHANGE_PATTERNS = (
    "no change",
    "no difference",
    "same as before",
    "scenes seem identical",
    "nothing has changed",
    "unchanged",
    "no observable difference",
    "no discernible difference",
    "remains the same",
    "without change",
)


class CaptionReference(StrictModel):
    item_id: str
    split: str
    references: List[str] = Field(min_length=1)
    change_flag: int = Field(ge=0, le=1)


class CaptionMetricRecord(StrictModel):
    item_id: str
    bleu_1: float
    bleu_4: float
    rouge_l: float
    change_flag_match: bool


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def caption_change_flag(text: str) -> int:
    value = " ".join(tokenize(text))
    return 0 if any(pattern in value for pattern in NO_CHANGE_PATTERNS) else 1


def load_reference_manifest(path: Path) -> Dict[str, CaptionReference]:
    records: Dict[str, CaptionReference] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = CaptionReference.model_validate(json.loads(line))
            if record.item_id in records:
                raise ValueError(
                    "duplicate item_id in reference manifest line {}: {}".format(
                        line_number, record.item_id
                    )
                )
            records[record.item_id] = record
    if not records:
        raise ValueError("reference manifest is empty")
    return records


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _closest_reference_length(lengths: Iterable[int], hypothesis_length: int) -> int:
    return min(lengths, key=lambda value: (abs(value - hypothesis_length), value))


def sentence_bleu(hypothesis: str, references: Sequence[str], n: int) -> float:
    hypothesis_tokens = tokenize(hypothesis)
    reference_tokens = [tokenize(reference) for reference in references]
    if not hypothesis_tokens or not reference_tokens:
        return 0.0
    precisions = []
    for order in range(1, n + 1):
        hypothesis_counts = _ngrams(hypothesis_tokens, order)
        total = sum(hypothesis_counts.values())
        if total == 0:
            return 0.0
        maximums: Counter = Counter()
        for tokens in reference_tokens:
            counts = _ngrams(tokens, order)
            for gram, count in counts.items():
                maximums[gram] = max(maximums[gram], count)
        clipped = sum(min(count, maximums[gram]) for gram, count in hypothesis_counts.items())
        if clipped == 0:
            return 0.0
        precisions.append(clipped / total)
    reference_length = _closest_reference_length(
        (len(tokens) for tokens in reference_tokens), len(hypothesis_tokens)
    )
    brevity = (
        1.0
        if len(hypothesis_tokens) > reference_length
        else math.exp(1.0 - reference_length / len(hypothesis_tokens))
    )
    return brevity * math.exp(sum(math.log(value) for value in precisions) / n)


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, start=1):
            current.append(
                previous[index - 1] + 1
                if left == right
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, references: Sequence[str], beta: float = 1.2) -> float:
    hypothesis_tokens = tokenize(hypothesis)
    if not hypothesis_tokens:
        return 0.0
    best = 0.0
    for reference in references:
        reference_tokens = tokenize(reference)
        if not reference_tokens:
            continue
        lcs = _lcs_length(reference_tokens, hypothesis_tokens)
        precision = lcs / len(hypothesis_tokens)
        recall = lcs / len(reference_tokens)
        if precision and recall:
            score = (
                (1 + beta**2) * precision * recall
                / (recall + beta**2 * precision)
            )
            best = max(best, score)
    return best


def evaluate_caption(
    item_id: str, caption: str, reference: CaptionReference
) -> CaptionMetricRecord:
    return CaptionMetricRecord(
        item_id=item_id,
        bleu_1=sentence_bleu(caption, reference.references, 1),
        bleu_4=sentence_bleu(caption, reference.references, 4),
        rouge_l=rouge_l(caption, reference.references),
        change_flag_match=caption_change_flag(caption) == reference.change_flag,
    )


def mean_caption_metrics(records: Sequence[CaptionMetricRecord]) -> Dict[str, float]:
    if not records:
        return {}
    size = len(records)
    return {
        "bleu_1": sum(record.bleu_1 for record in records) / size,
        "bleu_4": sum(record.bleu_4 for record in records) / size,
        "rouge_l": sum(record.rouge_l for record in records) / size,
        "change_flag_accuracy": sum(record.change_flag_match for record in records) / size,
    }


def references_from_levir_annotations(
    annotations_path: Path, split: str
) -> List[CaptionReference]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images", []) if isinstance(payload, dict) else []
    records = []
    for image in images:
        if not isinstance(image, dict) or image.get("split") != split:
            continue
        filename = str(image.get("filename", ""))
        item_id = Path(filename).stem
        references = [
            str(sentence.get("raw", "")).strip().strip(".").strip()
            for sentence in image.get("sentences", [])
            if isinstance(sentence, dict) and str(sentence.get("raw", "")).strip()
        ]
        records.append(
            CaptionReference(
                item_id=item_id,
                split=split,
                references=references,
                change_flag=int(image.get("changeflag", 0)),
            )
        )
    if not records:
        raise ValueError("no {} references found in {}".format(split, annotations_path))
    return records


def write_reference_manifest(records: Sequence[CaptionReference], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")
