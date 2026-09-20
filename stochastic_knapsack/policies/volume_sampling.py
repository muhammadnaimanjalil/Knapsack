"""Correlated volume-realization generation shared by Cases 2 and 3.

Sampling is kept outside the online policy so the decision rule is deterministic
for a supplied realization and arrival order, which improves reuse and testing.
"""

from __future__ import annotations

import numpy as np

from stochastic_knapsack.domain.exceptions import InputDataError
from stochastic_knapsack.domain.models import VolumeModel
from stochastic_knapsack.estimation.volume_estimator import (
    build_estimation_covariance,
)


def sample_volume_scenarios(
    model: VolumeModel,
    *,
    measurement_variance: float,
    rng: np.random.Generator,
    count: int = 1,
    max_batches: int = 100,
) -> np.ndarray:
    """Draw positive joint-Gaussian volume vectors by rejection sampling."""
    if count < 1:
        raise InputDataError("The number of volume realizations must be positive.")
    covariance = build_estimation_covariance(model, measurement_variance)
    accepted: list[np.ndarray] = []
    remaining = count
    for _ in range(max_batches):
        draws = rng.multivariate_normal(
            model.estimated_volumes,
            covariance,
            size=max(32, remaining * 2),
            check_valid="raise",
        )
        positive = draws[np.all(draws > 0.0, axis=1)]
        if positive.size:
            take = min(remaining, len(positive))
            accepted.append(positive[:take])
            remaining -= take
        if remaining == 0:
            return np.vstack(accepted)
    raise InputDataError(
        "Could not generate positive item volumes. Review the estimates or reduce variance."
    )
