"""Deterministic, dependency-light bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
import statistics
from typing import Dict, Mapping, Sequence

from pydantic import Field

from rs_agent.core.schemas import StrictModel

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260820
DEFAULT_CONFIDENCE = 0.95


class MeanConfidenceInterval(StrictModel):
    mean: float
    lower: float
    upper: float
    confidence: float = Field(gt=0.0, lt=1.0)
    sample_count: int = Field(ge=1)
    bootstrap_samples: int = Field(ge=1)
    seed: int


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> MeanConfidenceInterval:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    numeric = [float(value) for value in values]
    rng = random.Random(seed)
    size = len(numeric)
    means = [
        sum(numeric[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(bootstrap_samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return MeanConfidenceInterval(
        mean=statistics.fmean(numeric),
        lower=_quantile(means, alpha),
        upper=_quantile(means, 1.0 - alpha),
        confidence=confidence,
        sample_count=size,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def bootstrap_metric_intervals(
    metrics: Mapping[str, Sequence[float]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, MeanConfidenceInterval]:
    return {
        name: bootstrap_mean_interval(
            values,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
        for name, values in metrics.items()
        if values
    }
