"""Unit tests for the deterministic online decision engine."""

from stochastic_knapsack.application.case_2 import run_case_2
from stochastic_knapsack.domain.configurations import OnlineConfiguration


def test_online_policy_uses_realized_capacity_and_bid_price(small_problem):
    problem, model = small_problem
    result = run_case_2(
        problem, model,
        OnlineConfiguration(capacity=4, realization_variance=0.01, seed=7),
        realized_volumes=[2, 3, 1], arrival_order=[0, 1, 2],
    )
    assert [decision.accepted for decision in result.decisions] == [True, False, True]
    assert result.decisions[0].bid_price_per_liter == 2.0
    assert result.decisions[1].decision_reason == "insufficient_capacity"
    assert result.total_value == 14.0
    assert result.used_volume == 3.0


def test_generated_online_scenario_is_reproducible(small_problem):
    problem, model = small_problem
    configuration = OnlineConfiguration(capacity=4, realization_variance=0.01, seed=123)
    first = run_case_2(problem, model, configuration)
    second = run_case_2(problem, model, configuration)
    assert first.arrival_order == second.arrival_order
    assert first.realized_volumes == second.realized_volumes
    assert first.decisions == second.decisions
