"""Paired Monte Carlo comparison built from the reusable policy engines.

The offline portfolio is fixed once. Every run gives that portfolio and the
online policy the same volume vector, reducing comparison noise through common
random numbers. This module contains orchestration, not policy mathematics.
"""

from __future__ import annotations

import time

import numpy as np

from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.domain.results import (
    ComparisonStatistics,
    OfflineSolution,
    ScenarioComparison,
    SimulationResult,
)
from stochastic_knapsack.policies.online_bid_price import apply_online_policy
from stochastic_knapsack.policies.volume_sampling import sample_volume_scenarios
from stochastic_knapsack.simulation.statistics import summarize_policy


def _evaluate_offline(
    solution: OfflineSolution,
    problem: ProblemData,
    realized: np.ndarray,
    capacity: float,
) -> dict[str, float | int | bool]:
    selected_names = {item.name for item in solution.selected_items}
    indices = [i for i, item in enumerate(problem.items) if item.name in selected_names]
    used = float(np.sum(realized[indices]))
    overflow = max(0.0, used - capacity)
    return {
        "total_value": float(np.sum(problem.values[indices])),
        "item_count": len(indices),
        "used_volume": used,
        "capacity_utilization": used / capacity,
        "overflow_volume": overflow,
        "fit_without_overflow": overflow <= 1e-9,
    }


def compare_policies(
    problem: ProblemData,
    volume_model: VolumeModel,
    offline_solution: OfflineSolution,
    *,
    capacity: float,
    realization_variance: float,
    runs: int,
    seed: int,
) -> SimulationResult:
    """Evaluate the fixed offline solution and shared online policy over paired runs."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    volume_vectors = sample_volume_scenarios(
        volume_model,
        measurement_variance=realization_variance,
        rng=rng,
        count=runs,
    )
    offline_records: list[dict[str, float | int | bool]] = []
    online_records: list[dict[str, float | int | bool]] = []
    scenarios: list[ScenarioComparison] = []

    for run_index, realized in enumerate(volume_vectors, start=1):
        order = rng.permutation(len(problem.items))
        offline = _evaluate_offline(offline_solution, problem, realized, capacity)
        online_result = apply_online_policy(
            problem,
            volume_model,
            capacity=capacity,
            realized_volumes=realized,
            arrival_order=order,
            realization_variance=realization_variance,
            seed=seed + run_index,
        )
        online: dict[str, float | int | bool] = {
            "total_value": online_result.total_value,
            "item_count": online_result.accepted_item_count,
            "used_volume": online_result.used_volume,
            "capacity_utilization": online_result.capacity_utilization,
            "overflow_volume": 0.0,
            "fit_without_overflow": True,
        }
        offline_records.append(offline)
        online_records.append(online)
        scenarios.append(
            ScenarioComparison(
                run=run_index,
                offline_total_value=float(offline["total_value"]),
                offline_used_volume=float(offline["used_volume"]),
                offline_capacity_utilization=float(offline["capacity_utilization"]),
                offline_overflow_volume=float(offline["overflow_volume"]),
                offline_fit=bool(offline["fit_without_overflow"]),
                online_total_value=float(online["total_value"]),
                online_used_volume=float(online["used_volume"]),
                online_capacity_utilization=float(online["capacity_utilization"]),
                online_overflow_volume=0.0,
                online_fit=True,
            )
        )

    offline_stats = summarize_policy(offline_records)
    online_stats = summarize_policy(online_records)
    offline_values = np.asarray([row["total_value"] for row in offline_records], dtype=float)
    online_values = np.asarray([row["total_value"] for row in online_records], dtype=float)
    comparison = ComparisonStatistics(
        average_value_difference_online_minus_offline=float(np.mean(online_values - offline_values)),
        probability_online_value_exceeds_offline=float(np.mean(online_values > offline_values)),
        average_utilization_difference_online_minus_offline=float(
            online_stats.average_capacity_utilization - offline_stats.average_capacity_utilization
        ),
        average_overflow_reduction=float(
            offline_stats.average_overflow_volume - online_stats.average_overflow_volume
        ),
        fit_probability_improvement=float(
            online_stats.fit_probability - offline_stats.fit_probability
        ),
    )
    return SimulationResult(
        status="complete",
        method="paired Monte Carlo comparison",
        runs=runs,
        seed=seed,
        capacity=float(capacity),
        realization_variance=float(realization_variance),
        offline_solution=offline_solution,
        offline_statistics=offline_stats,
        online_statistics=online_stats,
        comparison=comparison,
        scenarios=tuple(scenarios),
        runtime_seconds=float(time.perf_counter() - started),
    )
