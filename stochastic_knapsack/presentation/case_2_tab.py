"""Streamlit view for one sequential online decision scenario."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from stochastic_knapsack.application.case_2 import run_case_2
from stochastic_knapsack.domain.configurations import OnlineConfiguration
from stochastic_knapsack.domain.exceptions import InputDataError
from stochastic_knapsack.presentation.components.common import (
    download_result, liters, percent, require_data, value_text,
)


LOGGER = logging.getLogger(__name__)
REASON_LABELS = {
    "accepted": "Accepted",
    "insufficient_capacity": "Rejected — insufficient capacity",
    "below_bid_price": "Rejected — value below opportunity cost",
}


def render_case_2_tab() -> None:
    """Render one online realization with its detailed trace initially collapsed."""
    st.markdown(
        """Items arrive one at a time. After an item's exact volume is revealed,
        the policy decides whether to accept it while reserving capacity for
        potentially valuable future items."""
    )
    left, right = st.columns(2)
    capacity = left.number_input(
        "Capacity (liters)", min_value=0.01, value=40.0, step=1.0,
        key="online_capacity", help="Reduced by each accepted item's actual volume."
    )
    seed = right.number_input(
        "Scenario seed", min_value=0, max_value=2_147_483_647, value=2026,
        step=1, key="online_seed", help="Reuse a seed to reproduce this scenario."
    )
    with st.expander("Advanced settings"):
        variance = st.number_input(
            "Volume-realization variance", min_value=0.0001, value=2.0, step=0.1,
            key="online_variance", help="Controls dispersion of correlated actual volumes."
        )
        st.caption("A fractional-knapsack relaxation estimates the opportunity cost of capacity.")

    if st.button("Generate arrivals and run policy", type="primary", width="stretch") and require_data():
        try:
            with st.spinner("Generating volumes and processing arrivals…"):
                st.session_state["online_result"] = run_case_2(
                    st.session_state["problem"], st.session_state["volume_model"],
                    OnlineConfiguration(
                        capacity=float(capacity), realization_variance=float(variance), seed=int(seed)
                    ),
                )
        except InputDataError as exc:
            st.error(str(exc), icon="⚠️")
        except Exception:
            LOGGER.exception("Unexpected Case 2 failure")
            st.error("An unexpected error occurred while running the online policy.")

    result = st.session_state.get("online_result")
    if result is None:
        st.info("Choose settings and select **Generate arrivals and run policy**.")
        return

    st.subheader("Online-policy result")
    metrics = st.columns(4)
    metrics[0].metric("Accepted value", value_text(result.total_value))
    metrics[1].metric("Items accepted", result.accepted_item_count)
    metrics[2].metric("Capacity used", liters(result.used_volume))
    metrics[3].metric("Capacity remaining", liters(result.remaining_capacity))
    st.dataframe(
        pd.DataFrame([{
            "Item": item.name, "Value": item.value,
            "Expected volume (L)": item.expected_volume,
            "Actual volume (L)": item.realized_volume,
        } for item in result.accepted_items]), hide_index=True, width="stretch"
    )
    st.caption("Next: open **Compare policies** to estimate long-run performance.")
    with st.expander("View all arrival decisions"):
        st.dataframe(pd.DataFrame([{
            "Arrival": row.arrival_position, "Item": row.item, "Value": row.value,
            "Actual volume (L)": row.realized_volume,
            "Bid price / L": row.bid_price_per_liter,
            "Opportunity cost": row.approximate_opportunity_cost,
            "Decision": REASON_LABELS[row.decision_reason],
            "Capacity remaining (L)": row.remaining_capacity_after,
        } for row in result.decisions]), hide_index=True, width="stretch")
    with st.expander("Technical details"):
        st.write({
            "Capacity utilization": percent(result.capacity_utilization),
            "DLP solves": result.dlp_solves, "Random seed": result.seed,
            "Runtime (seconds)": result.runtime_seconds,
        })
    download_result(result, "online_scenario_result.json", "download_online")
