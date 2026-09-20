"""Estimate item volumes from noisy historical package measurements.

The module owns OLS estimation and its diagnostics. It does not decide which
items to select or how volume realizations should be sampled.
"""

from __future__ import annotations

import math

import numpy as np

from stochastic_knapsack.domain.exceptions import InputDataError
from stochastic_knapsack.domain.models import ProblemData, VolumeModel


def estimate_volume_model(problem: ProblemData) -> VolumeModel:
    """Fit item volumes by OLS and verify statistical identifiability."""
    volumes, _, rank, singular_values = np.linalg.lstsq(
        problem.design_matrix, problem.measured_volumes, rcond=None
    )
    item_count = len(problem.items)
    if rank != item_count:
        raise InputDataError(
            "The historical packages do not uniquely identify every item volume: "
            f"matrix rank is {rank}, but {item_count} is required."
        )
    nonpositive = [
        problem.items[index].name
        for index, volume in enumerate(volumes)
        if volume <= 0
    ]
    if nonpositive:
        raise InputDataError(
            "The fitted model produced non-positive physical volumes for: "
            + ", ".join(nonpositive)
            + ". Add more informative package observations or review the data."
        )

    residuals = problem.measured_volumes - problem.design_matrix @ volumes
    degrees_of_freedom = problem.design_matrix.shape[0] - rank
    residual_variance = float(residuals @ residuals / degrees_of_freedom)
    information = problem.design_matrix.T @ problem.design_matrix
    information_inverse = np.linalg.inv(information)
    condition_number = float(singular_values[0] / singular_values[-1])
    return VolumeModel(
        estimated_volumes=volumes,
        information_inverse=information_inverse,
        residual_variance=residual_variance,
        residual_standard_deviation=math.sqrt(residual_variance),
        rank=int(rank),
        condition_number=condition_number,
        rmse=float(math.sqrt(np.mean(residuals**2))),
    )


def build_estimation_covariance(
    model: VolumeModel, measurement_variance: float
) -> np.ndarray:
    """Return the symmetric covariance matrix implied by a variance assumption."""
    if not math.isfinite(measurement_variance) or measurement_variance <= 0:
        raise InputDataError("Measurement variance must be a positive finite number.")
    covariance = measurement_variance * model.information_inverse
    return (covariance + covariance.T) / 2.0
