"""Integration tests proving that all use cases compose the shared engines."""

from stochastic_knapsack.application.case_3 import run_case_3
from stochastic_knapsack.domain.configurations import SimulationConfiguration


def test_paired_comparison_is_reproducible_and_online_never_overflows(small_problem):
    problem, model = small_problem
    configuration = SimulationConfiguration(
        capacity=4, confidence=0.95, realization_variance=0.01, runs=25, seed=123
    )
    first = run_case_3(problem, model, configuration)
    second = run_case_3(problem, model, configuration)
    assert first.scenarios == second.scenarios
    assert first.online_statistics.overflow_probability == 0.0
    assert first.offline_statistics.fit_probability + first.offline_statistics.overflow_probability == 1.0
