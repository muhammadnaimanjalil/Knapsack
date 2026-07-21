"""Statistical estimation and HiGHS optimization for Zalando's gift problem."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


class InputDataError(ValueError):
    """Raised when uploaded JSON does not match the required schema."""


class OptimizationError(RuntimeError):
    """Raised when HiGHS cannot produce a proven optimal solution."""


@dataclass(frozen=True)
class ProblemData:
    names: list[str]
    prices: np.ndarray
    design: np.ndarray
    measured_volumes: np.ndarray


@dataclass(frozen=True)
class RegressionResult:
    volumes: np.ndarray
    information_inverse: np.ndarray
    residual_variance: float
    residual_standard_deviation: float
    rank: int
    condition_number: float
    rmse: float


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputDataError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise InputDataError(f"{label} must be finite.")
    if nonnegative and number < 0:
        raise InputDataError(f"{label} cannot be negative.")
    return number


def validate_problem(items: Any, packages: Any) -> ProblemData:
    """Validate uploaded objects and construct the package incidence matrix."""
    if not isinstance(items, list) or not items:
        raise InputDataError("items.json must be a non-empty JSON array.")
    if not isinstance(packages, list) or not packages:
        raise InputDataError("packages.json must be a non-empty JSON array.")

    names: list[str] = []
    prices: list[float] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise InputDataError(f"Item {position + 1} must be a JSON object.")
        if "name" not in item or "price" not in item:
            raise InputDataError(
                f"Item {position + 1} must contain 'name' and 'price'."
            )
        name = item["name"]
        if not isinstance(name, str) or not name.strip():
            raise InputDataError(f"Item {position + 1} has an invalid name.")
        names.append(name.strip())
        prices.append(
            _finite_number(item["price"], f"Price for {name!r}", nonnegative=True)
        )

    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise InputDataError(f"Item names must be unique. Duplicates: {duplicates}")

    index = {name: i for i, name in enumerate(names)}
    design = np.zeros((len(packages), len(items)), dtype=float)
    measured = np.empty(len(packages), dtype=float)
    for row, package in enumerate(packages):
        label = f"Package {row + 1}"
        if not isinstance(package, dict):
            raise InputDataError(f"{label} must be a JSON object.")
        if "total_volume" not in package or "items" not in package:
            raise InputDataError(
                f"{label} must contain 'total_volume' and 'items'."
            )
        # Gaussian measurement noise is unbounded, so a small observed total can
        # be negative even though every underlying physical volume is positive.
        measured[row] = _finite_number(
            package["total_volume"], f"{label} total_volume"
        )
        package_items = package["items"]
        if not isinstance(package_items, list):
            raise InputDataError(f"{label} 'items' must be an array.")
        if len(package_items) != len(set(package_items)):
            raise InputDataError(f"{label} contains a repeated item.")
        for name in package_items:
            if not isinstance(name, str) or name not in index:
                raise InputDataError(f"{label} contains unknown item {name!r}.")
            design[row, index[name]] = 1.0

    if len(packages) <= len(items):
        raise InputDataError(
            "The regression needs more package observations than items. "
            f"Received {len(packages)} packages and {len(items)} items."
        )
    return ProblemData(names, np.asarray(prices), design, measured)


def estimate_volumes(problem: ProblemData) -> RegressionResult:
    """Estimate fixed item volumes with ordinary least squares."""
    volumes, _, rank, singular_values = np.linalg.lstsq(
        problem.design, problem.measured_volumes, rcond=None
    )
    item_count = len(problem.names)
    if rank != item_count:
        raise InputDataError(
            "The historical packages do not uniquely identify every item volume: "
            f"matrix rank is {rank}, but {item_count} is required."
        )
    nonpositive = [
        problem.names[i] for i, value in enumerate(volumes) if value <= 0
    ]
    if nonpositive:
        raise InputDataError(
            "The fitted model produced non-positive physical volumes for: "
            + ", ".join(nonpositive)
            + ". Add more informative package observations or review the data."
        )

    residuals = problem.measured_volumes - problem.design @ volumes
    degrees_of_freedom = problem.design.shape[0] - rank
    residual_variance = float(residuals @ residuals / degrees_of_freedom)
    information_inverse = np.linalg.inv(problem.design.T @ problem.design)
    condition_number = float(singular_values[0] / singular_values[-1])
    return RegressionResult(
        volumes=volumes,
        information_inverse=information_inverse,
        residual_variance=residual_variance,
        residual_standard_deviation=math.sqrt(residual_variance),
        rank=int(rank),
        condition_number=condition_number,
        rmse=float(math.sqrt(np.mean(residuals**2))),
    )


def _fit_probability(capacity: float, nominal: float, standard_error: float) -> float:
    if standard_error <= 0:
        return 1.0 if nominal <= capacity else 0.0
    return float(NormalDist().cdf((capacity - nominal) / standard_error))


def solve_chance_constrained(
    problem: ProblemData,
    regression: RegressionResult,
    *,
    capacity: float,
    confidence: float,
    noise_variance: float,
    time_limit: float = 60.0,
    max_cuts: int = 250,
    feasibility_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Solve the Gaussian chance-constrained binary knapsack with HiGHS.

    HiGHS solves a sequence of MILP outer approximations.  For a violated
    candidate x*, the norm's gradient creates the globally valid cut

        (v_hat + z * Sigma x* / sqrt(x*' Sigma x*))' x <= capacity.

    A candidate feasible for the original chance constraint and optimal for
    the relaxation is therefore globally optimal for the original binary model.
    """
    if capacity <= 0:
        raise InputDataError("Capacity must be positive.")
    if not 0.5 < confidence < 1.0:
        raise InputDataError("Confidence must be strictly between 50% and 100%.")
    if noise_variance <= 0 or not math.isfinite(noise_variance):
        raise InputDataError("Noise variance must be a positive finite number.")

    started = time.perf_counter()
    z_score = NormalDist().inv_cdf(confidence)
    covariance = noise_variance * regression.information_inverse
    item_count = len(problem.names)

    # The nominal capacity row is a valid initial relaxation because the
    # uncertainty buffer is non-negative.
    cut_rows: list[np.ndarray] = [regression.volumes.copy()]
    seen_candidates: set[tuple[int, ...]] = set()
    total_nodes = 0
    result = None
    selected_vector = None

    for iteration in range(1, max_cuts + 2):
        elapsed = time.perf_counter() - started
        remaining_time = time_limit - elapsed
        if remaining_time <= 0:
            raise OptimizationError(
                f"The optimization exceeded its {time_limit:g}-second time limit."
            )

        coefficients = np.vstack(cut_rows)
        constraint = LinearConstraint(
            coefficients,
            np.full(len(cut_rows), -np.inf),
            np.full(len(cut_rows), capacity),
        )
        result = milp(
            c=-problem.prices,
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
        total_nodes += int(getattr(result, "mip_node_count", 0) or 0)
        if result.status != 0 or result.x is None:
            raise OptimizationError(f"HiGHS did not prove optimality: {result.message}")

        selected_vector = np.rint(result.x).astype(float)
        if np.max(np.abs(result.x - selected_vector)) > 1e-5:
            raise OptimizationError("HiGHS returned a non-integral candidate.")

        nominal = float(regression.volumes @ selected_vector)
        variance = max(0.0, float(selected_vector @ covariance @ selected_vector))
        standard_error = math.sqrt(variance)
        chance_lhs = nominal + z_score * standard_error
        if chance_lhs <= capacity + feasibility_tolerance:
            break

        candidate_key = tuple(selected_vector.astype(int).tolist())
        if candidate_key in seen_candidates:
            raise OptimizationError(
                "The outer-approximation loop repeated a violated candidate. "
                "This indicates numerical instability in the uploaded data."
            )
        seen_candidates.add(candidate_key)
        if iteration > max_cuts:
            raise OptimizationError(
                f"The model exceeded the limit of {max_cuts} uncertainty cuts."
            )

        gradient = covariance @ selected_vector / standard_error
        cut_rows.append(regression.volumes + z_score * gradient)
    else:  # pragma: no cover - defensive; the loop raises at its configured cap.
        raise OptimizationError("The outer-approximation loop did not converge.")

    assert result is not None and selected_vector is not None
    selected_indices = np.flatnonzero(selected_vector > 0.5).tolist()
    uncertainty_buffer = z_score * standard_error
    total_price = float(problem.prices @ selected_vector)
    elapsed = time.perf_counter() - started
    selected_items = [
        {
            "name": problem.names[i],
            "price": float(problem.prices[i]),
            "estimated_volume": float(regression.volumes[i]),
        }
        for i in selected_indices
    ]

    all_items = [
        {
            "name": name,
            "price": float(problem.prices[i]),
            "estimated_volume": float(regression.volumes[i]),
            "selected": bool(selected_vector[i] > 0.5),
        }
        for i, name in enumerate(problem.names)
    ]

    return {
        "status": "optimal",
        "solver": "HiGHS via scipy.optimize.milp",
        "method": "exact MILP outer approximation of Gaussian chance constraint",
        "parameters": {
            "capacity": float(capacity),
            "confidence": float(confidence),
            "z_score": float(z_score),
            "noise_variance": float(noise_variance),
        },
        "objective": {"total_price": total_price},
        "totals": {
            "selected_item_count": len(selected_indices),
            "estimated_volume": nominal,
            "standard_error": standard_error,
            "uncertainty_buffer": uncertainty_buffer,
            "chance_constraint_lhs": chance_lhs,
            "capacity_slack": float(capacity - chance_lhs),
            "modeled_fit_probability": _fit_probability(
                capacity, nominal, standard_error
            ),
        },
        "selected_items": selected_items,
        "estimated_items": all_items,
        "diagnostics": {
            "item_count": item_count,
            "package_count": int(problem.design.shape[0]),
            "design_rank": regression.rank,
            "design_condition_number": regression.condition_number,
            "regression_rmse": regression.rmse,
            "fitted_residual_variance": regression.residual_variance,
            "fitted_residual_standard_deviation": regression.residual_standard_deviation,
            "outer_approximation_solves": iteration,
            "uncertainty_cuts_added": len(cut_rows) - 1,
            "highs_nodes": total_nodes,
            "mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
            "runtime_seconds": elapsed,
        },
    }
