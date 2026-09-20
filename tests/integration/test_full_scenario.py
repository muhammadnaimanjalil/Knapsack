"""Required end-to-end check using the committed repository scenario."""

from pathlib import Path

from stochastic_knapsack.application.case_1 import run_case_1
from stochastic_knapsack.application.common import prepare_problem
from stochastic_knapsack.data.json_loader import load_problem_paths
from stochastic_knapsack.domain.configurations import OfflineConfiguration


SCENARIO = Path(__file__).resolve().parents[1] / "Test Scenario"


def test_repository_scenario_reproduces_offline_optimum():
    items, packages = load_problem_paths(
        SCENARIO / "items.json", SCENARIO / "packages.json"
    )
    problem, model = prepare_problem(items, packages)
    result = run_case_1(problem, model, OfflineConfiguration())
    assert result.total_value == 735.0
    assert [item.name for item in result.selected_items] == [
        "A6", "A8", "A9", "A32", "A35", "A38", "A39", "A44", "A49"
    ]
    assert result.chance_constraint_lhs <= 40.0 + 1e-7
