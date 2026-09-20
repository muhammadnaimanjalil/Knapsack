"""Validated configuration objects for the three application cases."""

from __future__ import annotations

from dataclasses import dataclass
import math

from stochastic_knapsack.domain.exceptions import InputDataError


def _positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise InputDataError(f"{label} must be a positive finite number.")


@dataclass(frozen=True)
class OfflineConfiguration:
    capacity: float = 40.0
    confidence: float = 0.95
    measurement_variance: float = 2.0
    time_limit_seconds: float = 60.0
    max_cuts: int = 250

    def __post_init__(self) -> None:
        _positive(self.capacity, "Capacity")
        _positive(self.measurement_variance, "Measurement variance")
        _positive(self.time_limit_seconds, "Time limit")
        if not 0.5 < self.confidence < 1.0:
            raise InputDataError("Confidence must be strictly between 50% and 100%.")
        if self.max_cuts < 1:
            raise InputDataError("Maximum cuts must be positive.")


@dataclass(frozen=True)
class OnlineConfiguration:
    capacity: float = 40.0
    realization_variance: float = 2.0
    seed: int = 2026

    def __post_init__(self) -> None:
        _positive(self.capacity, "Capacity")
        _positive(self.realization_variance, "Realization variance")
        if self.seed < 0:
            raise InputDataError("Random seed cannot be negative.")


@dataclass(frozen=True)
class SimulationConfiguration:
    capacity: float = 40.0
    confidence: float = 0.95
    realization_variance: float = 2.0
    runs: int = 500
    seed: int = 2026
    offline_time_limit_seconds: float = 60.0

    def __post_init__(self) -> None:
        _positive(self.capacity, "Capacity")
        _positive(self.realization_variance, "Realization variance")
        _positive(self.offline_time_limit_seconds, "Offline time limit")
        if not 0.5 < self.confidence < 1.0:
            raise InputDataError("Confidence must be strictly between 50% and 100%.")
        if not 1 <= self.runs <= 10_000:
            raise InputDataError("Simulation runs must be between 1 and 10,000.")
        if self.seed < 0:
            raise InputDataError("Random seed cannot be negative.")
