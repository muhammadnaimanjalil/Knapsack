"""Policy-performance summaries independent of simulation generation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from stochastic_knapsack.domain.results import PolicyStatistics


def summarize_policy(records: Sequence[Mapping[str, float | int | bool]]) -> PolicyStatistics:
    """Calculate long-run value, utilization, and overflow statistics."""
    values = np.asarray([row["total_value"] for row in records], dtype=float)
    volumes = np.asarray([row["used_volume"] for row in records], dtype=float)
    utilization = np.asarray([row["capacity_utilization"] for row in records], dtype=float)
    overflow = np.asarray([row["overflow_volume"] for row in records], dtype=float)
    fit = np.asarray([row["fit_without_overflow"] for row in records], dtype=float)
    counts = np.asarray([row["item_count"] for row in records], dtype=float)
    sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    mean_value = float(np.mean(values))
    half_width = 1.96 * sample_std / math.sqrt(len(values))
    return PolicyStatistics(
        runs=len(records),
        average_total_value=mean_value,
        value_standard_deviation=sample_std,
        mean_value_95_percent_ci=(mean_value - half_width, mean_value + half_width),
        value_percentiles={
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        },
        average_used_volume=float(np.mean(volumes)),
        average_capacity_utilization=float(np.mean(utilization)),
        average_overflow_volume=float(np.mean(overflow)),
        fit_probability=float(np.mean(fit)),
        overflow_probability=float(1.0 - np.mean(fit)),
        average_item_count=float(np.mean(counts)),
    )
