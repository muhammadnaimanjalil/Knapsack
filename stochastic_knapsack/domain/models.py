"""Core problem and statistical-model representations.

The objects in this module are independent of Streamlit, JSON, and solver APIs.
They form the shared inputs used by the offline, online, and simulation cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Item:
    """One candidate item with a unique name and nonnegative value."""

    name: str
    value: float


@dataclass(frozen=True)
class PackageObservation:
    """One historical package composition and its measured total volume."""

    total_volume: float
    item_names: tuple[str, ...]


@dataclass(frozen=True)
class ProblemData:
    """Validated input data plus its numerical optimization representation."""

    items: tuple[Item, ...]
    packages: tuple[PackageObservation, ...]
    design_matrix: np.ndarray
    measured_volumes: np.ndarray

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.items)

    @property
    def values(self) -> np.ndarray:
        return np.asarray([item.value for item in self.items], dtype=float)

    # Compatibility aliases keep the mathematical notation readable.
    @property
    def prices(self) -> np.ndarray:
        return self.values

    @property
    def design(self) -> np.ndarray:
        return self.design_matrix


@dataclass(frozen=True)
class VolumeModel:
    """OLS volume estimates and diagnostics shared by all three cases."""

    estimated_volumes: np.ndarray
    information_inverse: np.ndarray
    residual_variance: float
    residual_standard_deviation: float
    rank: int
    condition_number: float
    rmse: float

    @property
    def volumes(self) -> np.ndarray:
        """Compatibility alias for the estimated-volume vector."""
        return self.estimated_volumes
