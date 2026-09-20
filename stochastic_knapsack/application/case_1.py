"""Application service for the fixed offline portfolio case."""

from stochastic_knapsack.domain.configurations import OfflineConfiguration
from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.domain.results import OfflineSolution
from stochastic_knapsack.optimization.offline_optimizer import solve_offline_problem


def run_case_1(
    problem: ProblemData,
    volume_model: VolumeModel,
    configuration: OfflineConfiguration,
) -> OfflineSolution:
    """Solve Case 1 through the reusable offline optimizer."""
    return solve_offline_problem(problem, volume_model, configuration)
