"""Small deterministic fixtures shared across unit and integration tests."""

import pytest

from stochastic_knapsack.application.common import prepare_problem


@pytest.fixture
def small_problem():
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
    return prepare_problem(items, packages)
