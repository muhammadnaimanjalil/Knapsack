"""Streamlit interface for the chance-constrained gift optimizer."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from optimizer import (
    InputDataError,
    OptimizationError,
    estimate_volumes,
    solve_chance_constrained,
    validate_problem,
)


st.set_page_config(
    page_title="Gift Portfolio Optimizer",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
      .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255, 105, 0, 0.13), rgba(255, 255, 255, 0.02));
        margin-bottom: 1.3rem;
      }
      .hero h1 {margin: 0; font-size: 2.15rem; letter-spacing: -0.035em;}
      .hero p {margin: 0.45rem 0 0; opacity: 0.78; font-size: 1.02rem;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 0.85rem 1rem;
        border-radius: 14px;
      }
      div[data-testid="stFileUploader"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 0.6rem 0.8rem 0.1rem;
      }
    </style>
    <div class="hero">
      <h1>Gift Portfolio Optimizer</h1>
      <p>Estimate missing item volumes from noisy package history, then find the highest-value selection that fits with your chosen confidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def parse_json_upload(uploaded_file: Any, label: str) -> Any:
    if uploaded_file is None:
        raise InputDataError(f"Please upload {label}.")
    try:
        return json.loads(uploaded_file.getvalue().decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise InputDataError(f"{label} must use UTF-8 encoding.") from exc
    except json.JSONDecodeError as exc:
        raise InputDataError(
            f"{label} is not valid JSON (line {exc.lineno}, column {exc.colno})."
        ) from exc


def money(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


with st.sidebar:
    st.header("Model settings")
    capacity = st.number_input(
        "Backpack capacity (L)", min_value=0.01, value=40.0, step=1.0
    )
    confidence_percent = st.slider(
        "Minimum fit confidence", min_value=50.1, max_value=99.9, value=95.0, step=0.1
    )
    noise_variance = st.number_input(
        "Measurement-error variance", min_value=0.0001, value=2.0, step=0.1,
        help="The problem statement specifies variance = 2."
    )
    time_limit = st.number_input(
        "Solver time limit (seconds)", min_value=5, max_value=600, value=60, step=5
    )
    st.divider()
    st.caption(
        "The optimizer uses ordinary least squares for item volumes and "
        "HiGHS MILP outer approximation for the Gaussian chance constraint."
    )


with st.form("optimization_form", clear_on_submit=False):
    left, right = st.columns(2, gap="large")
    with left:
        items_file = st.file_uploader(
            "Items data",
            type=["json"],
            key="items_file",
            help="JSON array with name and price for every item.",
        )
        st.caption('Expected fields: `[{"name": "A1", "price": 98}, ...]`')
    with right:
        packages_file = st.file_uploader(
            "Package history",
            type=["json"],
            key="packages_file",
            help="JSON array with measured total_volume and the items in each package.",
        )
        st.caption(
            'Expected fields: `[{"total_volume": 36.04, "items": ["A3", "A28"]}, ...]`'
        )

    submitted = st.form_submit_button(
        "Run optimization", type="primary", icon="▶", width="stretch"
    )


if submitted:
    st.session_state.pop("optimization_result", None)
    try:
        items_payload = parse_json_upload(items_file, "items.json")
        packages_payload = parse_json_upload(packages_file, "packages.json")
        with st.spinner("Estimating volumes and solving with HiGHS…"):
            problem = validate_problem(items_payload, packages_payload)
            regression = estimate_volumes(problem)
            result = solve_chance_constrained(
                problem,
                regression,
                capacity=float(capacity),
                confidence=float(confidence_percent / 100.0),
                noise_variance=float(noise_variance),
                time_limit=float(time_limit),
            )
        st.session_state["optimization_result"] = result
    except (InputDataError, OptimizationError) as exc:
        st.error(str(exc), icon="⚠️")
    except Exception as exc:  # Keep deployment errors readable without hiding details.
        st.exception(exc)


result = st.session_state.get("optimization_result")
if result:
    totals = result["totals"]
    diagnostics = result["diagnostics"]
    st.success("A globally optimal chance-feasible selection was found.", icon="✅")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total value", money(result["objective"]["total_price"]))
    metric_columns[1].metric("Selected items", totals["selected_item_count"])
    metric_columns[2].metric("Estimated volume", f"{totals['estimated_volume']:.3f} L")
    metric_columns[3].metric(
        "Chance-adjusted volume", f"{totals['chance_constraint_lhs']:.3f} L"
    )

    st.subheader("Selected items")
    selected_frame = pd.DataFrame(result["selected_items"]).rename(
        columns={
            "name": "Item",
            "price": "Price",
            "estimated_volume": "Estimated volume (L)",
        }
    )
    st.dataframe(
        selected_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Estimated volume (L)": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    detail_left, detail_right = st.columns(2, gap="large")
    with detail_left:
        st.subheader("Capacity and risk")
        risk_rows = pd.DataFrame(
            {
                "Measure": [
                    "Estimated volume",
                    f"{result['parameters']['confidence']:.1%} uncertainty buffer",
                    "Chance-constraint total",
                    "Capacity slack",
                    "Modeled fit probability",
                ],
                "Value": [
                    f"{totals['estimated_volume']:.6f} L",
                    f"{totals['uncertainty_buffer']:.6f} L",
                    f"{totals['chance_constraint_lhs']:.6f} L",
                    f"{totals['capacity_slack']:.6f} L",
                    f"{totals['modeled_fit_probability']:.4%}",
                ],
            }
        )
        st.dataframe(risk_rows, hide_index=True, width="stretch")
    with detail_right:
        st.subheader("Run diagnostics")
        diagnostic_rows = pd.DataFrame(
            {
                "Measure": [
                    "Items / packages",
                    "Regression rank",
                    "Fitted residual variance",
                    "Outer-approximation solves",
                    "HiGHS MIP gap",
                    "Runtime",
                ],
                "Value": [
                    f"{diagnostics['item_count']} / {diagnostics['package_count']}",
                    str(diagnostics["design_rank"]),
                    f"{diagnostics['fitted_residual_variance']:.6f}",
                    str(diagnostics["outer_approximation_solves"]),
                    f"{diagnostics['mip_gap']:.3%}",
                    f"{diagnostics['runtime_seconds']:.3f} s",
                ],
            }
        )
        st.dataframe(diagnostic_rows, hide_index=True, width="stretch")

    fitted_variance = diagnostics["fitted_residual_variance"]
    assumed_variance = result["parameters"]["noise_variance"]
    if abs(fitted_variance - assumed_variance) / assumed_variance > 0.5:
        st.warning(
            "The fitted residual variance differs substantially from the configured "
            f"measurement variance ({fitted_variance:.3f} vs {assumed_variance:.3f}). "
            "Consider rerunning with the fitted value as a sensitivity analysis."
        )

    with st.expander("View all estimated item volumes"):
        all_items_frame = pd.DataFrame(result["estimated_items"]).rename(
            columns={
                "name": "Item",
                "price": "Price",
                "estimated_volume": "Estimated volume (L)",
                "selected": "Selected",
            }
        )
        st.dataframe(all_items_frame, hide_index=True, width="stretch")

    st.download_button(
        "Download results as JSON",
        data=json.dumps(result, indent=2),
        file_name="gift_optimization_result.json",
        mime="application/json",
        type="primary",
        icon="⬇️",
        on_click="ignore",
        width="stretch",
    )
else:
    st.info(
        "Upload both JSON files, choose the model settings, and select **Run optimization**."
    )

with st.expander("How the model works"):
    st.markdown(
        """
        1. Package history is converted into an incidence matrix and used to estimate
           each item's volume by ordinary least squares.
        2. Estimation covariance is calculated from the configured Gaussian measurement variance.
        3. The app maximizes total price subject to a one-sided chance constraint.
        4. Because HiGHS solves MILPs rather than mixed-integer conic models, the app
           adds globally valid tangent cuts until the exact binary chance constraint is satisfied.
        """
    )
