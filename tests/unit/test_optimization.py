"""Unit tests for the two independent optimization components."""

import pytest

from stochastic_knapsack.domain.configurations import OfflineConfiguration
from stochastic_knapsack.optimization.deterministic_lp import solve_capacity_relaxation
from stochastic_knapsack.optimization.offline_optimizer import solve_offline_problem


def test_fractional_relaxation_returns_marginal_density():
    result = solve_capacity_relaxation([10, 6], [2, 2], capacity=3)
    assert result.objective_value == 13.0
    assert result.bid_price == 3.0
    assert result.allocation == (1.0, 0.5)


def test_offline_optimizer_returns_typed_feasible_solution(small_problem):
    problem, model = small_problem
    result = solve_offline_problem(
        problem, model,
        OfflineConfiguration(capacity=4, confidence=0.95, measurement_variance=0.01),
    )
    assert result.status == "optimal"
    assert result.chance_constraint_lhs <= 4.0 + 1e-7
    assert result.diagnostics.mip_gap == pytest.approx(0.0)
    assert result.to_dict()["selected_items"]
