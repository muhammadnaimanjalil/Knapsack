"""Unit tests for simulation statistics separated from orchestration."""

from stochastic_knapsack.simulation.statistics import summarize_policy


def test_single_run_statistics_have_finite_zero_width_interval():
    result = summarize_policy([{
        "total_value": 10.0, "item_count": 1, "used_volume": 2.0,
        "capacity_utilization": 0.5, "overflow_volume": 0.0,
        "fit_without_overflow": True,
    }])
    assert result.value_standard_deviation == 0.0
    assert result.mean_value_95_percent_ci == (10.0, 10.0)
    assert result.fit_probability == 1.0
