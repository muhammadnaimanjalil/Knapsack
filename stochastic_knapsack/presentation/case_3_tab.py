"""Streamlit view for paired long-run policy comparison."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from stochastic_knapsack.application.case_3 import run_case_3
from stochastic_knapsack.domain.configurations import SimulationConfiguration
from stochastic_knapsack.domain.exceptions import InputDataError, OptimizationError
from stochastic_knapsack.presentation.components.common import download_result, require_data


LOGGER = logging.getLogger(__name__)


def render_case_3_tab() -> None:
    """Show decision-relevant comparisons first and defer run-level detail."""
    st.markdown(
        """Run both policies over the same volume scenarios to compare their
        average value, capacity use, and overflow risk."""
    )
    left, right = st.columns(2)
    capacity = left.number_input(
        "Capacity (liters)", min_value=0.01, value=40.0, step=1.0,
        key="simulation_capacity", help="Common capacity used by both policies."
    )
    runs = right.number_input(
        "Simulation runs", min_value=10, max_value=10_000, value=500, step=100,
        key="simulation_runs", help="More runs reduce sampling noise but take longer."
    )
    with st.expander("Advanced settings"):
        confidence_percent = st.slider(
            "Offline fit confidence", min_value=50.1, max_value=99.9,
            value=95.0, step=0.1, key="simulation_confidence"
        )
        variance = st.number_input(
            "Volume-realization variance", min_value=0.0001, value=2.0,
            step=0.1, key="simulation_variance"
        )
        seed = st.number_input(
            "Simulation seed", min_value=0, max_value=2_147_483_647,
            value=2026, step=1, key="simulation_seed"
        )
        time_limit = st.number_input(
            "Offline solver time limit (seconds)", min_value=5, max_value=600,
            value=60, step=5, key="simulation_time_limit"
        )
        st.caption("Both policies receive the same correlated volume realization in each run.")

    if st.button("Run policy comparison", type="primary", width="stretch") and require_data():
        try:
            with st.spinner("Optimizing the benchmark and simulating paired scenarios…"):
                st.session_state["simulation_result"] = run_case_3(
                    st.session_state["problem"], st.session_state["volume_model"],
                    SimulationConfiguration(
                        capacity=float(capacity), confidence=float(confidence_percent / 100.0),
                        realization_variance=float(variance), runs=int(runs), seed=int(seed),
                        offline_time_limit_seconds=float(time_limit)
                    ),
                )
        except (InputDataError, OptimizationError) as exc:
            st.error(str(exc), icon="⚠️")
        except Exception:
            LOGGER.exception("Unexpected Case 3 failure")
            st.error("An unexpected error occurred while comparing the policies.")

    result = st.session_state.get("simulation_result")
    if result is None:
        st.info("Choose settings and select **Run policy comparison**.")
        return

    offline, online = result.offline_statistics, result.online_statistics
    st.subheader("Policy comparison")
    summary = pd.DataFrame([
        {"Measure": "Average value", "Offline portfolio": offline.average_total_value,
         "Online policy": online.average_total_value},
        {"Measure": "Average capacity utilization", "Offline portfolio": offline.average_capacity_utilization,
         "Online policy": online.average_capacity_utilization},
        {"Measure": "Probability of fitting", "Offline portfolio": offline.fit_probability,
         "Online policy": online.fit_probability},
        {"Measure": "Average overflow (L)", "Offline portfolio": offline.average_overflow_volume,
         "Online policy": online.average_overflow_volume},
    ])
    st.dataframe(summary, hide_index=True, width="stretch")
    with st.expander("Comparison details"):
        st.write(result.comparison.to_dict())
        st.write({
            "Offline 95% mean-value interval": offline.mean_value_95_percent_ci,
            "Online 95% mean-value interval": online.mean_value_95_percent_ci,
            "Simulation runtime (seconds)": result.runtime_seconds,
        })
    with st.expander("Explore individual simulation runs"):
        st.dataframe(pd.DataFrame([scenario.to_dict() for scenario in result.scenarios]),
                     hide_index=True, width="stretch")
    download_result(result, "policy_comparison_result.json", "download_simulation")
