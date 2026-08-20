"""Dependency-light corpus BLEU and CIDEr for supplementary caption evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Mapping, Sequence, Tuple

from rs_agent.evaluation.reference_metrics import CaptionReference, tokenize


def _ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def _closest_length(references: Sequence[Sequence[str]], hypothesis_size: int) -> int:
    return min(
        (len(reference) for reference in references),
        key=lambda size: (abs(size - hypothesis_size), size),
    )


def corpus_bleu(
    captions: Mapping[str, str],
    references: Mapping[str, CaptionReference],
    max_order: int = 4,
) -> Dict[str, float]:
    """Compute cumulative corpus BLEU-1 through BLEU-N without smoothing."""
    clipped = [0] * max_order
    totals = [0] * max_order
    hypothesis_length = 0
    reference_length = 0
    for item_id, caption in captions.items():
        reference = references.get(item_id)
        if reference is None:
            continue
        hypothesis = tokenize(caption)
        tokenized_references = [tokenize(text) for text in reference.references]
        if not hypothesis or not tokenized_references:
            continue
        hypothesis_length += len(hypothesis)
        reference_length += _closest_length(tokenized_references, len(hypothesis))
        for order in range(1, max_order + 1):
            counts = _ngrams(hypothesis, order)
            totals[order - 1] += sum(counts.values())
            maximums: Counter = Counter()
            for tokens in tokenized_references:
                for gram, count in _ngrams(tokens, order).items():
                    maximums[gram] = max(maximums[gram], count)
            clipped[order - 1] += sum(
                min(count, maximums[gram]) for gram, count in counts.items()
            )
    if not hypothesis_length:
        return {"corpus_bleu_{}".format(order): 0.0 for order in range(1, max_order + 1)}
    brevity = (
        1.0
        if hypothesis_length > reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    output = {}
    for maximum in range(1, max_order + 1):
        precisions = [
            clipped[index] / totals[index] if totals[index] else 0.0
            for index in range(maximum)
        ]
        score = (
            brevity
            * math.exp(sum(math.log(value) for value in precisions) / maximum)
            if all(precisions)
            else 0.0
        )
        output["corpus_bleu_{}".format(maximum)] = score
    return output


def _document_frequency(
    references: Mapping[str, CaptionReference], order: int
) -> Counter:
    frequency: Counter = Counter()
    for reference in references.values():
        grams = set()
        for text in reference.references:
            grams.update(_ngrams(tokenize(text), order))
        frequency.update(grams)
    return frequency


def _tf_idf_vector(
    text: str,
    order: int,
    frequency: Counter,
    document_count: int,
) -> Tuple[Dict[Tuple[str, ...], float], float]:
    counts = _ngrams(tokenize(text), order)
    vector = {}
    for gram, count in counts.items():
        document_frequency = max(1, frequency.get(gram, 0))
        vector[gram] = float(count) * math.log(document_count / document_frequency)
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return vector, norm


def _cosine(
    left: Mapping[Tuple[str, ...], float],
    left_norm: float,
    right: Mapping[Tuple[str, ...], float],
    right_norm: float,
) -> float:
    if not left_norm or not right_norm:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(gram, 0.0) for gram, value in left.items()) / (
        left_norm * right_norm
    )


def cider(
    captions: Mapping[str, str],
    references: Mapping[str, CaptionReference],
    max_order: int = 4,
) -> float:
    """Compute the standard TF-IDF cosine CIDEr score, scaled by 10."""
    document_count = max(1, len(references))
    frequencies = {
        order: _document_frequency(references, order)
        for order in range(1, max_order + 1)
    }
    item_scores = []
    for item_id, caption in captions.items():
        reference = references.get(item_id)
        if reference is None:
            continue
        order_scores = []
        for order in range(1, max_order + 1):
            candidate, candidate_norm = _tf_idf_vector(
                caption, order, frequencies[order], document_count
            )
            similarities = []
            for text in reference.references:
                target, target_norm = _tf_idf_vector(
                    text, order, frequencies[order], document_count
                )
                similarities.append(
                    _cosine(candidate, candidate_norm, target, target_norm)
                )
            order_scores.append(
                sum(similarities) / len(similarities) if similarities else 0.0
            )
        item_scores.append(sum(order_scores) / max_order)
    return 10.0 * sum(item_scores) / len(item_scores) if item_scores else 0.0


def corpus_caption_metrics(
    captions: Mapping[str, str],
    references: Mapping[str, CaptionReference],
) -> Dict[str, float]:
    output = corpus_bleu(captions, references)
    output["cider"] = cider(captions, references)
    return output
