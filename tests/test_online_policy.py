import numpy as np

from online_policy import (
    simulate_baseline_and_online,
    solve_deterministic_capacity_lp,
    solve_online_bid_price_scenario,
)
from optimizer import estimate_volumes, solve_chance_constrained, validate_problem


def _small_problem():
    items = [
        {"name": "A", "price": 10},
        {"name": "B", "price": 6},
        {"name": "C", "price": 4},
    ]
    packages = [
        {"total_volume": 2.0, "items": ["A"]},
        {"total_volume": 3.0, "items": ["B"]},
        {"total_volume": 1.0, "items": ["C"]},
        {"total_volume": 5.0, "items": ["A", "B"]},
        {"total_volume": 3.0, "items": ["A", "C"]},
        {"total_volume": 4.0, "items": ["B", "C"]},
        {"total_volume": 6.0, "items": ["A", "B", "C"]},
    ]
    problem = validate_problem(items, packages)
    return problem, estimate_volumes(problem)


def test_deterministic_lp_returns_marginal_density():
    result = solve_deterministic_capacity_lp([10, 6], [2, 2], capacity=3)

    assert result["objective_value"] == 13.0
    assert result["bid_price"] == 3.0
    assert result["allocation"] == [1.0, 0.5]
    assert result["capacity_used"] == 3.0


def test_online_policy_uses_realized_capacity_and_bid_price():
    problem, regression = _small_problem()
    result = solve_online_bid_price_scenario(
        problem,
        regression,
        capacity=4.0,
        noise_variance=0.01,
        seed=7,
        realized_volumes=[2.0, 3.0, 1.0],
        arrival_order=[0, 1, 2],
    )

    assert result["scenario"]["arrival_order"] == ["A", "B", "C"]
    assert [row["accepted"] for row in result["decisions"]] == [True, False, True]
    assert result["decisions"][0]["bid_price_per_liter"] == 2.0
    assert result["decisions"][1]["decision_reason"] == (
        "rejected_insufficient_capacity"
    )
    assert result["totals"]["total_value"] == 14.0
    assert result["totals"]["used_volume"] == 3.0
    assert result["totals"]["remaining_capacity"] == 1.0
    assert result["totals"]["overflow_volume"] == 0.0
    assert result["totals"]["fit_without_overflow"] is True


def test_paired_simulation_is_reproducible_and_reports_overflow_statistics():
    problem, regression = _small_problem()
    baseline = solve_chance_constrained(
        problem,
        regression,
        capacity=4.0,
        confidence=0.95,
        noise_variance=0.01,
    )

    first = simulate_baseline_and_online(
        problem,
        regression,
        baseline,
        capacity=4.0,
        noise_variance=0.01,
        runs=25,
        seed=123,
    )
    second = simulate_baseline_and_online(
        problem,
        regression,
        baseline,
        capacity=4.0,
        noise_variance=0.01,
        runs=25,
        seed=123,
    )

    assert first["scenario_results"] == second["scenario_results"]
    assert first["case_1_baseline"] == second["case_1_baseline"]
    assert first["case_2_online"] == second["case_2_online"]
    assert first["case_2_online"]["overflow_probability"] == 0.0
    assert first["case_2_online"]["fit_probability"] == 1.0
    assert (
        first["case_1_baseline"]["fit_probability"]
        + first["case_1_baseline"]["overflow_probability"]
        == 1.0
    )
    assert np.isfinite(
        first["comparison"]["average_value_difference_case_2_minus_case_1"]
    )
