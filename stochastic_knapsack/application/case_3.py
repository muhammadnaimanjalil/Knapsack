"""Application service for the paired offline-versus-online comparison."""

from stochastic_knapsack.application.case_1 import run_case_1
from stochastic_knapsack.domain.configurations import (
    OfflineConfiguration,
    SimulationConfiguration,
)
from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.domain.results import SimulationResult
from stochastic_knapsack.simulation.policy_comparison import compare_policies


def run_case_3(
    problem: ProblemData,
    volume_model: VolumeModel,
    configuration: SimulationConfiguration,
) -> SimulationResult:
    """Solve the offline benchmark once, then compare both policies fairly."""
    offline = run_case_1(
        problem,
        volume_model,
        OfflineConfiguration(
            capacity=configuration.capacity,
            confidence=configuration.confidence,
            measurement_variance=configuration.realization_variance,
            time_limit_seconds=configuration.offline_time_limit_seconds,
        ),
    )
    return compare_policies(
        problem,
        volume_model,
        offline,
        capacity=configuration.capacity,
        realization_variance=configuration.realization_variance,
        runs=configuration.runs,
        seed=configuration.seed,
    )
