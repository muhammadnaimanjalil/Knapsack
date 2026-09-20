"""Compatibility imports for the refactored online and simulation packages.

The calculation source of truth now lives under ``stochastic_knapsack``. This
facade intentionally contains no policy or simulation implementation.
"""

from stochastic_knapsack.application.case_2 import run_case_2
from stochastic_knapsack.application.case_3 import run_case_3
from stochastic_knapsack.estimation.volume_estimator import (
    build_estimation_covariance as estimation_covariance,
)
from stochastic_knapsack.optimization.deterministic_lp import (
    solve_capacity_relaxation,
)
from stochastic_knapsack.policies.online_bid_price import apply_online_policy
from stochastic_knapsack.policies.volume_sampling import (
    sample_volume_scenarios,
)
from stochastic_knapsack.simulation.policy_comparison import compare_policies

__all__ = [
    "apply_online_policy",
    "compare_policies",
    "estimation_covariance",
    "run_case_2",
    "run_case_3",
    "sample_volume_scenarios",
    "solve_capacity_relaxation",
]
