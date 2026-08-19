import pytest

from rs_agent.evaluation.bootstrap import (
    bootstrap_mean_interval,
    bootstrap_metric_intervals,
)


def test_bootstrap_mean_interval_is_deterministic_and_bounded() -> None:
    first = bootstrap_mean_interval([0.0, 1.0, 2.0, 3.0], bootstrap_samples=200, seed=7)
    second = bootstrap_mean_interval([0.0, 1.0, 2.0, 3.0], bootstrap_samples=200, seed=7)
    assert first == second
    assert first.mean == pytest.approx(1.5)
    assert first.lower <= first.mean <= first.upper
    assert first.sample_count == 4


def test_bootstrap_metric_intervals_handles_boolean_accuracy() -> None:
    intervals = bootstrap_metric_intervals(
        {"accuracy": [1.0, 0.0, 1.0]}, bootstrap_samples=100, seed=11
    )
    assert intervals["accuracy"].mean == pytest.approx(2 / 3)


def test_bootstrap_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_mean_interval([])
    with pytest.raises(ValueError, match="positive"):
        bootstrap_mean_interval([1.0], bootstrap_samples=0)
