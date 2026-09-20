"""Unit tests for transport parsing and shared domain validation."""

import pytest

from stochastic_knapsack.data.json_loader import load_json_bytes
from stochastic_knapsack.data.validation import validate_problem
from stochastic_knapsack.domain.exceptions import InputDataError


def test_json_loader_reports_source_and_location():
    with pytest.raises(InputDataError, match=r"items.json.*line 1"):
        load_json_bytes(b'[{"name":}]', "items.json")


def test_validation_rejects_duplicate_item_names():
    items = [{"name": "A", "price": 1}, {"name": "A", "price": 2}]
    packages = [{"total_volume": 1, "items": ["A"]}] * 3
    with pytest.raises(InputDataError, match="unique"):
        validate_problem(items, packages)


def test_validation_builds_shared_problem_model(small_problem):
    problem, model = small_problem
    assert problem.design_matrix.shape == (7, 3)
    assert problem.names == ("A", "B", "C")
    assert model.rank == 3
