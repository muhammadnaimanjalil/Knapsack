"""Exact fractional-knapsack relaxation used to calculate capacity bid prices.

The one-resource LP is solved by value density rather than a general solver.
It is deterministic and independent of arrivals, sampling, and presentation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from stochastic_knapsack.domain.exceptions import InputDataError
from stochastic_knapsack.domain.results import CapacityRelaxationSolution


TOLERANCE = 1e-9


def solve_capacity_relaxation(
    values: Iterable[float], expected_volumes: Iterable[float], capacity: float
) -> CapacityRelaxationSolution:
    """Solve the fractional knapsack and return the marginal value of capacity."""
    value_array = np.asarray(list(values), dtype=float)
    volume_array = np.asarray(list(expected_volumes), dtype=float)
    if value_array.shape != volume_array.shape:
        raise InputDataError("Values and expected volumes must have equal lengths.")
    if not math.isfinite(capacity) or capacity < 0:
        raise InputDataError("Remaining capacity must be finite and nonnegative.")
    if np.any(~np.isfinite(value_array)) or np.any(value_array < 0):
        raise InputDataError("Future item values must be finite and nonnegative.")
    if np.any(~np.isfinite(volume_array)) or np.any(volume_array <= 0):
        raise InputDataError("Expected future item volumes must be positive.")

    item_count = len(value_array)
    if item_count == 0:
        return CapacityRelaxationSolution(0.0, 0.0, (), 0.0)

    densities = value_array / volume_array
    order = np.argsort(-densities, kind="stable")
    allocation = np.zeros(item_count, dtype=float)
    remaining = float(capacity)
    objective = 0.0
    capacity_used = 0.0
    bid_price = float(densities[order[0]]) if capacity <= TOLERANCE else 0.0

    if capacity > float(np.sum(volume_array)) + TOLERANCE:
        return CapacityRelaxationSolution(
            float(np.sum(value_array)),
            0.0,
            tuple(np.ones(item_count, dtype=float).tolist()),
            float(np.sum(volume_array)),
        )

    for index in order:
        if remaining <= TOLERANCE:
            break
        volume = float(volume_array[index])
        fraction = min(1.0, remaining / volume)
        allocation[index] = fraction
        used = fraction * volume
        objective += fraction * float(value_array[index])
        capacity_used += used
        remaining -= used
        # The density of the marginal included item is the LP's left derivative.
        bid_price = float(densities[index])
        if fraction < 1.0 - TOLERANCE:
            break

    return CapacityRelaxationSolution(
        float(objective), bid_price, tuple(allocation.tolist()), float(capacity_used)
    )
