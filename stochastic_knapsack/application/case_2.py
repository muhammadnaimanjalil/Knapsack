"""Application service for generating and evaluating one online scenario."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from stochastic_knapsack.domain.configurations import OnlineConfiguration
from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.domain.results import OnlineScenarioResult
from stochastic_knapsack.policies.online_bid_price import apply_online_policy
from stochastic_knapsack.policies.volume_sampling import sample_volume_scenarios


def run_case_2(
    problem: ProblemData,
    volume_model: VolumeModel,
    configuration: OnlineConfiguration,
    *,
    realized_volumes: Iterable[float] | None = None,
    arrival_order: Iterable[int] | None = None,
) -> OnlineScenarioResult:
    """Generate missing scenario inputs and apply the shared online policy."""
    rng = np.random.default_rng(configuration.seed)
    realized = (
        sample_volume_scenarios(
            volume_model,
            measurement_variance=configuration.realization_variance,
            rng=rng,
            count=1,
        )[0]
        if realized_volumes is None
        else np.asarray(list(realized_volumes), dtype=float)
    )
    order = (
        rng.permutation(len(problem.items))
        if arrival_order is None
        else np.asarray(list(arrival_order), dtype=int)
    )
    return apply_online_policy(
        problem,
        volume_model,
        capacity=configuration.capacity,
        realized_volumes=realized,
        arrival_order=order,
        realization_variance=configuration.realization_variance,
        seed=configuration.seed,
    )
