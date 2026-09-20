"""Offline Gaussian chance-constrained knapsack optimization.

HiGHS solves successive MILP outer approximations of the covariance-norm
constraint. This module receives already validated and estimated data; it does
not parse files, estimate volumes, simulate scenarios, or render results.
"""

from __future__ import annotations

import math
import time
from statistics import NormalDist

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from stochastic_knapsack.domain.configurations import OfflineConfiguration
from stochastic_knapsack.domain.exceptions import OptimizationError
from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.domain.results import (
    EstimatedItem,
    OfflineDiagnostics,
    OfflineSolution,
)
from stochastic_knapsack.estimation.volume_estimator import (
    build_estimation_covariance,
)


def _fit_probability(capacity: float, nominal: float, standard_error: float) -> float:
    if standard_error <= 0:
        return 1.0 if nominal <= capacity else 0.0
    return float(NormalDist().cdf((capacity - nominal) / standard_error))


def solve_offline_problem(
    problem: ProblemData,
    volume_model: VolumeModel,
    configuration: OfflineConfiguration,
    *,
    feasibility_tolerance: float = 1e-7,
) -> OfflineSolution:
    """Return a globally optimal fixed portfolio under the Gaussian constraint."""
    started = time.perf_counter()
    z_score = NormalDist().inv_cdf(configuration.confidence)
    covariance = build_estimation_covariance(
        volume_model, configuration.measurement_variance
    )
    item_count = len(problem.items)

    # The nominal constraint is a valid relaxation because the uncertainty
    # buffer is nonnegative for every binary selection.
    cut_rows: list[np.ndarray] = [volume_model.estimated_volumes.copy()]
    seen_candidates: set[tuple[int, ...]] = set()
    total_nodes = 0
    solver_result = None
    selected_vector = None

    for iteration in range(1, configuration.max_cuts + 2):
        remaining_time = configuration.time_limit_seconds - (
            time.perf_counter() - started
        )
        if remaining_time <= 0:
            raise OptimizationError(
                f"Optimization exceeded its {configuration.time_limit_seconds:g}-second limit."
            )
        rows = np.vstack(cut_rows)
        constraint = LinearConstraint(
            rows,
            np.full(len(cut_rows), -np.inf),
            np.full(len(cut_rows), configuration.capacity),
        )
        solver_result = milp(
            c=-problem.values,
            integrality=np.ones(item_count, dtype=np.uint8),
            bounds=Bounds(np.zeros(item_count), np.ones(item_count)),
            constraints=constraint,
            options={
                "disp": False,
                "presolve": True,
                "time_limit": remaining_time,
                "mip_rel_gap": 0.0,
            },
        )
        total_nodes += int(getattr(solver_result, "mip_node_count", 0) or 0)
        if solver_result.status != 0 or solver_result.x is None:
            raise OptimizationError(
                f"HiGHS did not prove optimality: {solver_result.message}"
            )

        selected_vector = np.rint(solver_result.x).astype(float)
        if np.max(np.abs(solver_result.x - selected_vector)) > 1e-5:
            raise OptimizationError("HiGHS returned a non-integral candidate.")

        nominal = float(volume_model.estimated_volumes @ selected_vector)
        variance = max(0.0, float(selected_vector @ covariance @ selected_vector))
        standard_error = math.sqrt(variance)
        chance_lhs = nominal + z_score * standard_error
        if chance_lhs <= configuration.capacity + feasibility_tolerance:
            break

        candidate = tuple(selected_vector.astype(int).tolist())
        if candidate in seen_candidates:
            raise OptimizationError(
                "Outer approximation repeated a violated candidate; check numerical conditioning."
            )
        seen_candidates.add(candidate)
        if iteration > configuration.max_cuts:
            raise OptimizationError(
                f"The model exceeded {configuration.max_cuts} uncertainty cuts."
            )

        # Convexity makes this tangent a globally valid relaxation cut. It
        # removes the current violation without removing a feasible portfolio.
        gradient = covariance @ selected_vector / standard_error
        cut_rows.append(volume_model.estimated_volumes + z_score * gradient)
    else:  # pragma: no cover - defensive loop guard
        raise OptimizationError("Outer approximation did not converge.")

    assert solver_result is not None and selected_vector is not None
    selected = selected_vector > 0.5
    items = tuple(
        EstimatedItem(
            item.name,
            float(item.value),
            float(volume_model.estimated_volumes[index]),
            bool(selected[index]),
        )
        for index, item in enumerate(problem.items)
    )
    elapsed = time.perf_counter() - started
    return OfflineSolution(
        status="optimal",
        solver="HiGHS via scipy.optimize.milp",
        method="exact MILP outer approximation of Gaussian chance constraint",
        capacity=float(configuration.capacity),
        confidence=float(configuration.confidence),
        z_score=float(z_score),
        measurement_variance=float(configuration.measurement_variance),
        total_value=float(problem.values @ selected_vector),
        estimated_volume=nominal,
        standard_error=standard_error,
        uncertainty_buffer=float(z_score * standard_error),
        chance_constraint_lhs=chance_lhs,
        capacity_slack=float(configuration.capacity - chance_lhs),
        modeled_fit_probability=_fit_probability(
            configuration.capacity, nominal, standard_error
        ),
        selected_items=tuple(item for item in items if item.selected),
        estimated_items=items,
        diagnostics=OfflineDiagnostics(
            item_count=item_count,
            package_count=len(problem.packages),
            design_rank=volume_model.rank,
            design_condition_number=volume_model.condition_number,
            regression_rmse=volume_model.rmse,
            fitted_residual_variance=volume_model.residual_variance,
            fitted_residual_standard_deviation=volume_model.residual_standard_deviation,
            outer_approximation_solves=iteration,
            uncertainty_cuts_added=len(cut_rows) - 1,
            highs_nodes=total_nodes,
            mip_gap=float(getattr(solver_result, "mip_gap", 0.0) or 0.0),
            runtime_seconds=float(elapsed),
        ),
    )
