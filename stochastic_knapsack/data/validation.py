"""Convert untrusted JSON values into validated stochastic-knapsack data.

All three cases share this module. It checks the public file schema, enforces
domain constraints, and constructs the package-item incidence matrix used by
volume estimation. It performs no regression or optimization.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from stochastic_knapsack.domain.exceptions import InputDataError
from stochastic_knapsack.domain.models import Item, PackageObservation, ProblemData


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputDataError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise InputDataError(f"{label} must be finite.")
    if nonnegative and number < 0:
        raise InputDataError(f"{label} cannot be negative.")
    return number


def validate_problem(items_payload: Any, packages_payload: Any) -> ProblemData:
    """Validate both input documents and build a reusable ``ProblemData`` object."""
    if not isinstance(items_payload, list) or not items_payload:
        raise InputDataError("items.json must be a non-empty JSON array.")
    if not isinstance(packages_payload, list) or not packages_payload:
        raise InputDataError("packages.json must be a non-empty JSON array.")

    items: list[Item] = []
    seen_names: set[str] = set()
    for position, raw_item in enumerate(items_payload, start=1):
        if not isinstance(raw_item, dict):
            raise InputDataError(f"Item {position} must be a JSON object.")
        if "name" not in raw_item or "price" not in raw_item:
            raise InputDataError(f"Item {position} must contain 'name' and 'price'.")
        raw_name = raw_item["name"]
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise InputDataError(f"Item {position} has an invalid name.")
        name = raw_name.strip()
        if name in seen_names:
            raise InputDataError(f"Item names must be unique. Duplicate: {name!r}.")
        seen_names.add(name)
        items.append(
            Item(
                name=name,
                value=_finite_number(
                    raw_item["price"], f"Value for {name!r}", nonnegative=True
                ),
            )
        )

    index = {item.name: position for position, item in enumerate(items)}
    design = np.zeros((len(packages_payload), len(items)), dtype=float)
    measured = np.empty(len(packages_payload), dtype=float)
    packages: list[PackageObservation] = []

    for row, raw_package in enumerate(packages_payload):
        label = f"Package {row + 1}"
        if not isinstance(raw_package, dict):
            raise InputDataError(f"{label} must be a JSON object.")
        if "total_volume" not in raw_package or "items" not in raw_package:
            raise InputDataError(f"{label} must contain 'total_volume' and 'items'.")
        total_volume = _finite_number(
            raw_package["total_volume"], f"{label} total_volume"
        )
        raw_names = raw_package["items"]
        if not isinstance(raw_names, list):
            raise InputDataError(f"{label} 'items' must be an array.")
        if any(not isinstance(name, str) for name in raw_names):
            raise InputDataError(f"{label} item names must be strings.")
        if len(raw_names) != len(set(raw_names)):
            raise InputDataError(f"{label} contains a repeated item.")
        for name in raw_names:
            if name not in index:
                raise InputDataError(f"{label} contains unknown item {name!r}.")
            design[row, index[name]] = 1.0
        measured[row] = total_volume
        packages.append(PackageObservation(total_volume, tuple(raw_names)))

    if len(packages) <= len(items):
        raise InputDataError(
            "The regression needs more package observations than items. "
            f"Received {len(packages)} packages and {len(items)} items."
        )
    return ProblemData(tuple(items), tuple(packages), design, measured)
