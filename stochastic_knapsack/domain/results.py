"""Typed results passed between calculation, simulation, and UI layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class SerializableResult:
    """Mixin providing JSON-ready dictionaries for download and presentation."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EstimatedItem(SerializableResult):
    name: str
    value: float
    estimated_volume: float
    selected: bool


@dataclass(frozen=True)
class OfflineDiagnostics(SerializableResult):
    item_count: int
    package_count: int
    design_rank: int
    design_condition_number: float
    regression_rmse: float
    fitted_residual_variance: float
    fitted_residual_standard_deviation: float
    outer_approximation_solves: int
    uncertainty_cuts_added: int
    highs_nodes: int
    mip_gap: float
    runtime_seconds: float


@dataclass(frozen=True)
class OfflineSolution(SerializableResult):
    status: str
    solver: str
    method: str
    capacity: float
    confidence: float
    z_score: float
    measurement_variance: float
    total_value: float
    estimated_volume: float
    standard_error: float
    uncertainty_buffer: float
    chance_constraint_lhs: float
    capacity_slack: float
    modeled_fit_probability: float
    selected_items: tuple[EstimatedItem, ...]
    estimated_items: tuple[EstimatedItem, ...]
    diagnostics: OfflineDiagnostics


@dataclass(frozen=True)
class CapacityRelaxationSolution(SerializableResult):
    objective_value: float
    bid_price: float
    allocation: tuple[float, ...]
    capacity_used: float


@dataclass(frozen=True)
class OnlineDecision(SerializableResult):
    arrival_position: int
    item: str
    value: float
    expected_volume: float
    realized_volume: float
    remaining_capacity_before: float
    bid_price_per_liter: float
    approximate_opportunity_cost: float
    approximate_marginal_value: float
    future_dlp_value: float
    fits_remaining_capacity: bool
    accepted: bool
    decision_reason: str
    remaining_capacity_after: float
    cumulative_value: float


@dataclass(frozen=True)
class AcceptedItem(SerializableResult):
    name: str
    value: float
    expected_volume: float
    realized_volume: float


@dataclass(frozen=True)
class OnlineScenarioResult(SerializableResult):
    status: str
    method: str
    capacity: float
    realization_variance: float
    seed: int
    arrival_order: tuple[str, ...]
    realized_volumes: dict[str, float]
    total_value: float
    accepted_item_count: int
    rejected_item_count: int
    used_volume: float
    remaining_capacity: float
    capacity_utilization: float
    accepted_items: tuple[AcceptedItem, ...]
    decisions: tuple[OnlineDecision, ...]
    dlp_solves: int
    runtime_seconds: float


@dataclass(frozen=True)
class PolicyStatistics(SerializableResult):
    runs: int
    average_total_value: float
    value_standard_deviation: float
    mean_value_95_percent_ci: tuple[float, float]
    value_percentiles: dict[str, float]
    average_used_volume: float
    average_capacity_utilization: float
    average_overflow_volume: float
    fit_probability: float
    overflow_probability: float
    average_item_count: float


@dataclass(frozen=True)
class ComparisonStatistics(SerializableResult):
    average_value_difference_online_minus_offline: float
    probability_online_value_exceeds_offline: float
    average_utilization_difference_online_minus_offline: float
    average_overflow_reduction: float
    fit_probability_improvement: float


@dataclass(frozen=True)
class ScenarioComparison(SerializableResult):
    run: int
    offline_total_value: float
    offline_used_volume: float
    offline_capacity_utilization: float
    offline_overflow_volume: float
    offline_fit: bool
    online_total_value: float
    online_used_volume: float
    online_capacity_utilization: float
    online_overflow_volume: float
    online_fit: bool


@dataclass(frozen=True)
class SimulationResult(SerializableResult):
    status: str
    method: str
    runs: int
    seed: int
    capacity: float
    realization_variance: float
    offline_solution: OfflineSolution
    offline_statistics: PolicyStatistics
    online_statistics: PolicyStatistics
    comparison: ComparisonStatistics
    scenarios: tuple[ScenarioComparison, ...]
    runtime_seconds: float
