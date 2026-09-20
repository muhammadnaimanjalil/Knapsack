"""Shared data-preparation service used before running any case."""

from __future__ import annotations

from typing import Any

from stochastic_knapsack.data.validation import validate_problem
from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.estimation.volume_estimator import estimate_volume_model


def prepare_problem(
    items_payload: Any, packages_payload: Any
) -> tuple[ProblemData, VolumeModel]:
    """Validate raw payloads once and estimate the shared item-volume model."""
    problem = validate_problem(items_payload, packages_payload)
    return problem, estimate_volume_model(problem)
