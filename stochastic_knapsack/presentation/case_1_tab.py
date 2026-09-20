"""Streamlit view for the offline portfolio; calculations live elsewhere."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from stochastic_knapsack.application.case_1 import run_case_1
from stochastic_knapsack.domain.configurations import OfflineConfiguration
from stochastic_knapsack.domain.exceptions import InputDataError, OptimizationError
from stochastic_knapsack.presentation.components.common import (
    download_result,
    liters,
    percent,
    require_data,
    value_text,
)


LOGGER = logging.getLogger(__name__)


def render_case_1_tab() -> None:
    """Render essential settings first and hide scientific diagnostics by default."""
    st.markdown(
        """Select a fixed portfolio before exact item volumes are known. The
        optimizer balances value against volume uncertainty and the required
        probability of fitting."""
    )
    left, right = st.columns(2)
    capacity = left.number_input(
        "Capacity (liters)", min_value=0.01, value=40.0, step=1.0,
        key="offline_capacity", help="Total volume available to the fixed portfolio."
    )
    confidence_percent = right.slider(
        "Required fit confidence", min_value=50.1, max_value=99.9, value=95.0,
        step=0.1, key="offline_confidence",
        help="Minimum modeled probability that the selected portfolio fits."
    )
    with st.expander("Advanced settings"):
        variance = st.number_input(
            "Measurement-error variance", min_value=0.0001, value=2.0, step=0.1,
            key="offline_variance", help="Controls uncertainty in estimated item volumes."
        )
        time_limit = st.number_input(
            "Solver time limit (seconds)", min_value=5, max_value=600, value=60,
            step=5, key="offline_time_limit"
        )
        st.caption("A Gaussian chance constraint is solved through HiGHS outer approximation.")

    if st.button("Optimize portfolio", type="primary", width="stretch") and require_data():
        try:
            with st.spinner("Estimating risk and optimizing the portfolio…"):
                st.session_state["offline_result"] = run_case_1(
                    st.session_state["problem"], st.session_state["volume_model"],
                    OfflineConfiguration(
                        capacity=float(capacity), confidence=float(confidence_percent / 100.0),
                        measurement_variance=float(variance), time_limit_seconds=float(time_limit)
                    ),
                )
        except (InputDataError, OptimizationError) as exc:
            st.error(str(exc), icon="⚠️")
        except Exception:
            LOGGER.exception("Unexpected Case 1 failure")
            st.error("An unexpected error occurred while optimizing the portfolio.")

    result = st.session_state.get("offline_result")
    if result is None:
        st.info("Choose settings and select **Optimize portfolio** to see results.")
        return

    st.subheader("Portfolio result")
    metrics = st.columns(4)
    metrics[0].metric("Portfolio value", value_text(result.total_value))
    metrics[1].metric("Items selected", len(result.selected_items))
    metrics[2].metric("Estimated volume", liters(result.estimated_volume))
    metrics[3].metric("Fit probability", percent(result.modeled_fit_probability))
    st.dataframe(
        pd.DataFrame([{
            "Item": item.name, "Value": item.value,
            "Estimated volume (L)": item.estimated_volume,
        } for item in result.selected_items]), hide_index=True, width="stretch"
    )
    st.caption("Next: open **Online decisions** to see how an adaptive policy responds to one realization.")
    with st.expander("Optimization diagnostics"):
        st.write({
            "Uncertainty buffer (L)": result.uncertainty_buffer,
            "Chance-constraint total (L)": result.chance_constraint_lhs,
            "Capacity slack (L)": result.capacity_slack,
            "Regression RMSE": result.diagnostics.regression_rmse,
            "Design condition number": result.diagnostics.design_condition_number,
            "Outer-approximation solves": result.diagnostics.outer_approximation_solves,
            "Uncertainty cuts": result.diagnostics.uncertainty_cuts_added,
            "HiGHS MIP gap": result.diagnostics.mip_gap,
            "Runtime (seconds)": result.diagnostics.runtime_seconds,
        })
    with st.expander("View all estimated item volumes"):
        st.dataframe(pd.DataFrame([item.to_dict() for item in result.estimated_items]),
                     hide_index=True, width="stretch")
    download_result(result, "offline_portfolio_result.json", "download_offline")
