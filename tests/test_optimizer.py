import json
from pathlib import Path

import numpy as np
import pytest

from optimizer import estimate_volumes, solve_chance_constrained, validate_problem


DATA_DIR = Path(__file__).resolve().parents[1] / ".codex_doc_review"


def test_original_dataset_reproduces_optimum():
    if not (DATA_DIR / "items.json").exists():
        pytest.skip("Original challenge datasets are not committed to the repository")
    items = json.loads((DATA_DIR / "items.json").read_text(encoding="utf-8"))
    packages = json.loads((DATA_DIR / "packages.json").read_text(encoding="utf-8"))
    problem = validate_problem(items, packages)
    regression = estimate_volumes(problem)
    result = solve_chance_constrained(
        problem,
        regression,
        capacity=40.0,
        confidence=0.95,
        noise_variance=2.0,
    )

    assert result["status"] == "optimal"
    assert result["objective"]["total_price"] == 735.0
    assert [item["name"] for item in result["selected_items"]] == [
        "A6",
        "A8",
        "A9",
        "A32",
        "A35",
        "A38",
        "A39",
        "A44",
        "A49",
    ]
    assert result["totals"]["chance_constraint_lhs"] <= 40.0 + 1e-7
    assert result["diagnostics"]["mip_gap"] == 0.0


def test_small_identifiable_problem():
    items = [{"name": "A", "price": 8}, {"name": "B", "price": 7}]
    packages = [
        {"total_volume": 2.0, "items": ["A"]},
        {"total_volume": 3.0, "items": ["B"]},
        {"total_volume": 5.0, "items": ["A", "B"]},
        {"total_volume": 2.1, "items": ["A"]},
    ]
    problem = validate_problem(items, packages)
    regression = estimate_volumes(problem)
    result = solve_chance_constrained(
        problem,
        regression,
        capacity=4.0,
        confidence=0.95,
        noise_variance=0.01,
    )

    assert result["objective"]["total_price"] == 8.0
    assert result["selected_items"][0]["name"] == "A"
    assert np.isfinite(result["totals"]["modeled_fit_probability"])
