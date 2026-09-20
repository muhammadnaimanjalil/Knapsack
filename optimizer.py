"""Compatibility imports for the refactored offline calculation package.

New code should import from ``stochastic_knapsack``. This module remains small
so existing notebooks can migrate without retaining a second implementation.
"""

from stochastic_knapsack.data.validation import validate_problem
from stochastic_knapsack.domain.configurations import OfflineConfiguration
from stochastic_knapsack.domain.exceptions import InputDataError, OptimizationError
from stochastic_knapsack.domain.models import ProblemData, VolumeModel
from stochastic_knapsack.estimation.volume_estimator import estimate_volume_model
from stochastic_knapsack.optimization.offline_optimizer import solve_offline_problem


# Historical type/function names retained as explicit aliases.
RegressionResult = VolumeModel
estimate_volumes = estimate_volume_model

__all__ = [
    "InputDataError",
    "OfflineConfiguration",
    "OptimizationError",
    "ProblemData",
    "RegressionResult",
    "estimate_volumes",
    "solve_offline_problem",
    "validate_problem",
]
