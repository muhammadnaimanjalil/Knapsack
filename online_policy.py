"""Online bid-price control and paired simulation for the gift problem."""

from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np

from optimizer import InputDataError, ProblemData, RegressionResult


_TOLERANCE = 1e-9


def estimation_covariance(
    regression: RegressionResult, noise_variance: float
) -> np.ndarray:
    """Return the covariance used for item-volume realizations."""
    if noise_variance <= 0 or not math.isfinite(noise_variance):
        raise InputDataError("Noise variance must be a positive finite number.")
    covariance = noise_variance * regression.information_inverse
    return (covariance + covariance.T) / 2.0


def sample_positive_volume_vectors(
    regression: RegressionResult,
    *,
    noise_variance: float,
    rng: np.random.Generator,
    count: int = 1,
    max_batches: int = 100,
) -> np.ndarray:
    """Draw positive joint Gaussian item-volume vectors by rejection sampling."""
    if count < 1:
        raise InputDataError("The number of volume realizations must be positive.")

    covariance = estimation_covariance(regression, noise_variance)
    accepted: list[np.ndarray] = []
    remaining = count
    for _ in range(max_batches):
        batch_size = max(32, remaining * 2)
        draws = rng.multivariate_normal(
            regression.volumes,
            covariance,
            size=batch_size,
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
        "Could not generate positive item volumes from the fitted Gaussian model. "
        "Review the volume estimates or reduce the configured noise variance."
    )


def solve_deterministic_capacity_lp(
    prices: Iterable[float],
    expected_volumes: Iterable[float],
    capacity: float,
) -> dict[str, Any]:
    """Solve the one-resource deterministic LP and return its capacity bid price.

    The LP is a fractional knapsack:

        max p'y
        s.t. v'y <= capacity
             0 <= y <= 1.

    Its greedy density solution is exact.  The bid price is the left marginal
    value of capacity at the current solution, which is the density of the
    marginal accepted item.
    """
    price_array = np.asarray(list(prices), dtype=float)
    volume_array = np.asarray(list(expected_volumes), dtype=float)
    if price_array.shape != volume_array.shape:
        raise InputDataError("Prices and expected volumes must have equal lengths.")
    if capacity < 0 or not math.isfinite(capacity):
        raise InputDataError("Remaining capacity must be finite and nonnegative.")
    if np.any(~np.isfinite(price_array)) or np.any(price_array < 0):
        raise InputDataError("Future item prices must be finite and nonnegative.")
    if np.any(~np.isfinite(volume_array)) or np.any(volume_array <= 0):
        raise InputDataError("Expected future item volumes must be positive.")

    item_count = len(price_array)
    if item_count == 0:
        return {
            "objective_value": 0.0,
            "bid_price": 0.0,
            "allocation": [],
            "capacity_used": 0.0,
        }

    densities = price_array / volume_array
    order = np.argsort(-densities, kind="stable")
    allocation = np.zeros(item_count, dtype=float)
    remaining = float(capacity)
    objective = 0.0
    capacity_used = 0.0
    bid_price = float(densities[order[0]]) if capacity <= _TOLERANCE else 0.0
    total_volume = float(np.sum(volume_array))

    if capacity > total_volume + _TOLERANCE:
        allocation[:] = 1.0
        return {
            "objective_value": float(np.sum(price_array)),
            "bid_price": 0.0,
            "allocation": allocation.tolist(),
            "capacity_used": total_volume,
        }

    for index in order:
        if remaining <= _TOLERANCE:
            break
        volume = float(volume_array[index])
        fraction = min(1.0, remaining / volume)
        allocation[index] = fraction
        used = fraction * volume
        objective += fraction * float(price_array[index])
        capacity_used += used
        remaining -= used
        bid_price = float(densities[index])
        if fraction < 1.0 - _TOLERANCE:
            break

    return {
        "objective_value": float(objective),
        "bid_price": float(bid_price),
        "allocation": allocation.tolist(),
        "capacity_used": float(capacity_used),
    }


def _validated_order(
    arrival_order: Iterable[int] | None,
    item_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if arrival_order is None:
        return rng.permutation(item_count)
    order = np.asarray(list(arrival_order), dtype=int)
    if len(order) != item_count or sorted(order.tolist()) != list(range(item_count)):
        raise InputDataError("Arrival order must be a permutation of all item indices.")
    return order


def _validated_realizations(
    realized_volumes: Iterable[float] | None,
    regression: RegressionResult,
    noise_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if realized_volumes is None:
        return sample_positive_volume_vectors(
            regression,
            noise_variance=noise_variance,
            rng=rng,
            count=1,
        )[0]
    realized = np.asarray(list(realized_volumes), dtype=float)
    if realized.shape != regression.volumes.shape:
        raise InputDataError("Realized volumes must contain one value per item.")
    if np.any(~np.isfinite(realized)) or np.any(realized <= 0):
        raise InputDataError("Every realized item volume must be positive and finite.")
    return realized


def solve_online_bid_price_scenario(
    problem: ProblemData,
    regression: RegressionResult,
    *,
    capacity: float,
    noise_variance: float,
    seed: int,
    realized_volumes: Iterable[float] | None = None,
    arrival_order: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Run one sequential accept/reject scenario using reoptimized LP bid prices."""
    if capacity <= 0 or not math.isfinite(capacity):
        raise InputDataError("Capacity must be a positive finite number.")

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    item_count = len(problem.names)
    realized = _validated_realizations(
        realized_volumes, regression, noise_variance, rng
    )
    order = _validated_order(arrival_order, item_count, rng)

    remaining_capacity = float(capacity)
    total_value = 0.0
    used_volume = 0.0
    accepted_indices: list[int] = []
    decisions: list[dict[str, Any]] = []
    dlp_solves = 0

    for position, item_index in enumerate(order, start=1):
        future_indices = order[position:]
        dlp = solve_deterministic_capacity_lp(
            problem.prices[future_indices],
            regression.volumes[future_indices],
            remaining_capacity,
        )
        dlp_solves += 1

        price = float(problem.prices[item_index])
        expected_volume = float(regression.volumes[item_index])
        realized_volume = float(realized[item_index])
        bid_price = float(dlp["bid_price"])
        opportunity_cost = bid_price * realized_volume
        marginal_value = price - opportunity_cost
        fits = realized_volume <= remaining_capacity + _TOLERANCE
        accepted = bool(fits and marginal_value >= -_TOLERANCE)
        before = remaining_capacity

        if accepted:
            remaining_capacity = max(0.0, remaining_capacity - realized_volume)
            used_volume += realized_volume
            total_value += price
            accepted_indices.append(int(item_index))
            reason = "accepted_by_bid_price"
        elif not fits:
            reason = "rejected_insufficient_capacity"
        else:
            reason = "rejected_below_bid_price"

        decisions.append(
            {
                "arrival_position": position,
                "item": problem.names[item_index],
                "price": price,
                "expected_volume": expected_volume,
                "realized_volume": realized_volume,
                "remaining_capacity_before": float(before),
                "bid_price_per_liter": bid_price,
                "approximate_opportunity_cost": float(opportunity_cost),
                "approximate_marginal_value": float(marginal_value),
                "future_dlp_value": float(dlp["objective_value"]),
                "fits_remaining_capacity": bool(fits),
                "accepted": accepted,
                "decision_reason": reason,
                "remaining_capacity_after": float(remaining_capacity),
                "cumulative_value": float(total_value),
            }
        )

    elapsed = time.perf_counter() - started
    accepted_items = [
        {
            "name": problem.names[index],
            "price": float(problem.prices[index]),
            "expected_volume": float(regression.volumes[index]),
            "realized_volume": float(realized[index]),
        }
        for index in accepted_indices
    ]
    utilization = used_volume / capacity

    return {
        "status": "complete",
        "method": "approximate dynamic programming with reoptimized DLP bid prices",
        "parameters": {
            "capacity": float(capacity),
            "noise_variance": float(noise_variance),
            "seed": int(seed),
            "arrival_model": "uniform random permutation without replacement",
            "volume_model": "positive joint Gaussian rejection sample",
        },
        "scenario": {
            "arrival_order": [problem.names[index] for index in order],
            "realized_volumes": {
                problem.names[index]: float(realized[index])
                for index in range(item_count)
            },
        },
        "totals": {
            "total_value": float(total_value),
            "total_cost": float(total_value),
            "accepted_item_count": len(accepted_indices),
            "rejected_item_count": item_count - len(accepted_indices),
            "used_volume": float(used_volume),
            "remaining_capacity": float(remaining_capacity),
            "capacity_utilization": float(utilization),
            "overflow_volume": 0.0,
            "fit_without_overflow": True,
        },
        "accepted_items": accepted_items,
        "decisions": decisions,
        "diagnostics": {
            "item_count": item_count,
            "dlp_solves": dlp_solves,
            "runtime_seconds": float(elapsed),
        },
    }


def evaluate_baseline_realization(
    baseline_result: dict[str, Any],
    problem: ProblemData,
    realized_volumes: Iterable[float],
    *,
    capacity: float,
) -> dict[str, Any]:
    """Evaluate the fixed Case 1 portfolio under one realized volume vector."""
    realized = np.asarray(list(realized_volumes), dtype=float)
    selected_names = {item["name"] for item in baseline_result["selected_items"]}
    selected_indices = [
        index for index, name in enumerate(problem.names) if name in selected_names
    ]
    used_volume = float(np.sum(realized[selected_indices]))
    total_value = float(np.sum(problem.prices[selected_indices]))
    overflow = max(0.0, used_volume - capacity)
    return {
        "total_value": total_value,
        "total_cost": total_value,
        "selected_item_count": len(selected_indices),
        "used_volume": used_volume,
        "capacity_utilization": used_volume / capacity,
        "overflow_volume": overflow,
        "fit_without_overflow": bool(overflow <= _TOLERANCE),
    }


def _summarize_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["total_value"] for row in records], dtype=float)
    volumes = np.asarray([row["used_volume"] for row in records], dtype=float)
    utilization = np.asarray(
        [row["capacity_utilization"] for row in records], dtype=float
    )
    overflow = np.asarray([row["overflow_volume"] for row in records], dtype=float)
    fit = np.asarray([row["fit_without_overflow"] for row in records], dtype=float)
    counts = np.asarray(
        [
            row.get("accepted_item_count", row.get("selected_item_count", 0))
            for row in records
        ],
        dtype=float,
    )
    sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half_width = 1.96 * sample_std / math.sqrt(len(values))
    mean_value = float(np.mean(values))

    return {
        "runs": len(records),
        "average_total_value": mean_value,
        "average_total_cost": mean_value,
        "value_standard_deviation": sample_std,
        "mean_value_95_percent_ci": [
            float(mean_value - half_width),
            float(mean_value + half_width),
        ],
        "value_percentiles": {
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        },
        "average_used_volume": float(np.mean(volumes)),
        "average_capacity_utilization": float(np.mean(utilization)),
        "average_overflow_volume": float(np.mean(overflow)),
        "fit_probability": float(np.mean(fit)),
        "overflow_probability": float(1.0 - np.mean(fit)),
        "average_item_count": float(np.mean(counts)),
    }


def simulate_baseline_and_online(
    problem: ProblemData,
    regression: RegressionResult,
    baseline_result: dict[str, Any],
    *,
    capacity: float,
    noise_variance: float,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    """Run paired scenarios for the fixed baseline and online bid-price policy."""
    if runs < 1 or runs > 10_000:
        raise InputDataError("Simulation runs must be between 1 and 10,000.")
    if capacity <= 0 or not math.isfinite(capacity):
        raise InputDataError("Capacity must be a positive finite number.")

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    volume_vectors = sample_positive_volume_vectors(
        regression,
        noise_variance=noise_variance,
        rng=rng,
        count=runs,
    )
    baseline_records: list[dict[str, Any]] = []
    online_records: list[dict[str, Any]] = []
    scenario_records: list[dict[str, Any]] = []

    for run_index, realized in enumerate(volume_vectors, start=1):
        order = rng.permutation(len(problem.names))
        baseline = evaluate_baseline_realization(
            baseline_result,
            problem,
            realized,
            capacity=capacity,
        )
        online_result = solve_online_bid_price_scenario(
            problem,
            regression,
            capacity=capacity,
            noise_variance=noise_variance,
            seed=seed + run_index,
            realized_volumes=realized,
            arrival_order=order,
        )
        online = {
            "total_value": online_result["totals"]["total_value"],
            "total_cost": online_result["totals"]["total_cost"],
            "accepted_item_count": online_result["totals"]["accepted_item_count"],
            "used_volume": online_result["totals"]["used_volume"],
            "capacity_utilization": online_result["totals"][
                "capacity_utilization"
            ],
            "overflow_volume": online_result["totals"]["overflow_volume"],
            "fit_without_overflow": online_result["totals"][
                "fit_without_overflow"
            ],
        }
        baseline_records.append(baseline)
        online_records.append(online)
        scenario_records.append(
            {
                "run": run_index,
                "case_1_total_value": baseline["total_value"],
                "case_1_used_volume": baseline["used_volume"],
                "case_1_capacity_utilization": baseline[
                    "capacity_utilization"
                ],
                "case_1_overflow_volume": baseline["overflow_volume"],
                "case_1_fit": baseline["fit_without_overflow"],
                "case_2_total_value": online["total_value"],
                "case_2_used_volume": online["used_volume"],
                "case_2_capacity_utilization": online["capacity_utilization"],
                "case_2_overflow_volume": online["overflow_volume"],
                "case_2_fit": online["fit_without_overflow"],
            }
        )

    baseline_summary = _summarize_policy(baseline_records)
    online_summary = _summarize_policy(online_records)
    baseline_values = np.asarray(
        [row["total_value"] for row in baseline_records], dtype=float
    )
    online_values = np.asarray(
        [row["total_value"] for row in online_records], dtype=float
    )

    return {
        "status": "complete",
        "method": "paired Monte Carlo comparison",
        "parameters": {
            "runs": int(runs),
            "seed": int(seed),
            "capacity": float(capacity),
            "noise_variance": float(noise_variance),
            "common_random_numbers": True,
            "volume_model": "positive joint Gaussian rejection sample",
            "arrival_model": "uniform random permutation without replacement",
        },
        "case_1_baseline": baseline_summary,
        "case_2_online": online_summary,
        "comparison": {
            "average_value_difference_case_2_minus_case_1": float(
                np.mean(online_values - baseline_values)
            ),
            "probability_case_2_value_exceeds_case_1": float(
                np.mean(online_values > baseline_values)
            ),
            "average_utilization_difference_case_2_minus_case_1": float(
                online_summary["average_capacity_utilization"]
                - baseline_summary["average_capacity_utilization"]
            ),
            "average_overflow_reduction": float(
                baseline_summary["average_overflow_volume"]
                - online_summary["average_overflow_volume"]
            ),
            "fit_probability_improvement": float(
                online_summary["fit_probability"]
                - baseline_summary["fit_probability"]
            ),
        },
        "scenario_results": scenario_records,
        "diagnostics": {
            "runtime_seconds": float(time.perf_counter() - started),
            "item_count": len(problem.names),
        },
    }
