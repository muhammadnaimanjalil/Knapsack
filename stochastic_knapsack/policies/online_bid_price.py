"""Reusable sequential bid-price policy for a fully specified scenario.

The caller supplies exact realized volumes and an arrival permutation. At each
arrival, the policy re-solves the deterministic future-capacity relaxation and
accepts an item only if it fits and covers the opportunity cost of capacity.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

import numpy as np

from stochastic_knapsack.domain.exceptions import InputDataError
from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.domain.results import (
    AcceptedItem,
    OnlineDecision,
    OnlineScenarioResult,
)
from stochastic_knapsack.optimization.deterministic_lp import (
    TOLERANCE,
    solve_capacity_relaxation,
)


def _validate_scenario(
    realized_volumes: Iterable[float],
    arrival_order: Iterable[int],
    item_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    realized = np.asarray(list(realized_volumes), dtype=float)
    order = np.asarray(list(arrival_order), dtype=int)
    if realized.shape != (item_count,):
        raise InputDataError("Realized volumes must contain one value per item.")
    if np.any(~np.isfinite(realized)) or np.any(realized <= 0):
        raise InputDataError("Every realized volume must be positive and finite.")
    if len(order) != item_count or sorted(order.tolist()) != list(range(item_count)):
        raise InputDataError("Arrival order must be a permutation of all item indices.")
    return realized, order


def apply_online_policy(
    problem: ProblemData,
    volume_model: VolumeModel,
    *,
    capacity: float,
    realized_volumes: Iterable[float],
    arrival_order: Iterable[int],
    realization_variance: float,
    seed: int,
) -> OnlineScenarioResult:
    """Apply the online rule to one deterministic scenario and return its trace."""
    if not math.isfinite(capacity) or capacity <= 0:
        raise InputDataError("Capacity must be a positive finite number.")
    started = time.perf_counter()
    realized, order = _validate_scenario(
        realized_volumes, arrival_order, len(problem.items)
    )
    remaining = float(capacity)
    total_value = 0.0
    used_volume = 0.0
    accepted_indices: list[int] = []
    decisions: list[OnlineDecision] = []

    for position, item_index in enumerate(order, start=1):
        future_indices = order[position:]
        relaxation = solve_capacity_relaxation(
            problem.values[future_indices],
            volume_model.estimated_volumes[future_indices],
            remaining,
        )
        item = problem.items[item_index]
        realized_volume = float(realized[item_index])
        opportunity_cost = relaxation.bid_price * realized_volume
        marginal_value = item.value - opportunity_cost
        fits = realized_volume <= remaining + TOLERANCE
        accepted = bool(fits and marginal_value >= -TOLERANCE)
        before = remaining
        if accepted:
            remaining = max(0.0, remaining - realized_volume)
            used_volume += realized_volume
            total_value += item.value
            accepted_indices.append(int(item_index))
            reason = "accepted"
        elif not fits:
            reason = "insufficient_capacity"
        else:
            reason = "below_bid_price"
        decisions.append(
            OnlineDecision(
                arrival_position=position,
                item=item.name,
                value=float(item.value),
                expected_volume=float(volume_model.estimated_volumes[item_index]),
                realized_volume=realized_volume,
                remaining_capacity_before=float(before),
                bid_price_per_liter=relaxation.bid_price,
                approximate_opportunity_cost=float(opportunity_cost),
                approximate_marginal_value=float(marginal_value),
                future_dlp_value=relaxation.objective_value,
                fits_remaining_capacity=bool(fits),
                accepted=accepted,
                decision_reason=reason,
                remaining_capacity_after=float(remaining),
                cumulative_value=float(total_value),
            )
        )

    accepted_items = tuple(
        AcceptedItem(
            problem.items[index].name,
            float(problem.items[index].value),
            float(volume_model.estimated_volumes[index]),
            float(realized[index]),
        )
        for index in accepted_indices
    )
    return OnlineScenarioResult(
        status="complete",
        method="approximate dynamic programming with reoptimized DLP bid prices",
        capacity=float(capacity),
        realization_variance=float(realization_variance),
        seed=int(seed),
        arrival_order=tuple(problem.items[index].name for index in order),
        realized_volumes={
            item.name: float(realized[index])
            for index, item in enumerate(problem.items)
        },
        total_value=float(total_value),
        accepted_item_count=len(accepted_indices),
        rejected_item_count=len(problem.items) - len(accepted_indices),
        used_volume=float(used_volume),
        remaining_capacity=float(remaining),
        capacity_utilization=float(used_volume / capacity),
        accepted_items=accepted_items,
        decisions=tuple(decisions),
        dlp_solves=len(problem.items),
        runtime_seconds=float(time.perf_counter() - started),
    )
