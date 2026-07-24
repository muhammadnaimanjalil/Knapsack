"""Streamlit interface for baseline, online, and simulation gift models."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from online_policy import (
    simulate_baseline_and_online,
    solve_online_bid_price_scenario,
)
from optimizer import (
    InputDataError,
    OptimizationError,
    estimate_volumes,
    solve_chance_constrained,
    validate_problem,
)


st.set_page_config(
    page_title="Gift Portfolio Decision Lab",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem;}
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
      div[data-baseweb="tab-list"] {gap: 0.45rem;}
      button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
      }
    </style>
    <div class="hero">
      <h1>Gift Portfolio Decision Lab</h1>
      <p>Compare a chance-constrained portfolio with sequential bid-price control under uncertain item volumes.</p>
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


def load_model(items_file: Any, packages_file: Any) -> tuple[Any, Any]:
    items_payload = parse_json_upload(items_file, "items.json")
    packages_payload = parse_json_upload(packages_file, "packages.json")
    problem = validate_problem(items_payload, packages_payload)
    return problem, estimate_volumes(problem)


def money(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def json_download(label: str, result: dict[str, Any], file_name: str, key: str) -> None:
    st.download_button(
        label,
        data=json.dumps(result, indent=2),
        file_name=file_name,
        mime="application/json",
        type="primary",
        icon="⬇️",
        on_click="ignore",
        width="stretch",
        key=key,
    )


def render_case_1(result: dict[str, Any]) -> None:
    totals = result["totals"]
    diagnostics = result["diagnostics"]
    st.success("A globally optimal chance-feasible portfolio was found.", icon="✅")

    metrics = st.columns(5)
    metrics[0].metric("Portfolio value", money(result["objective"]["total_price"]))
    metrics[1].metric("Selected items", totals["selected_item_count"])
    metrics[2].metric("Estimated volume", f"{totals['estimated_volume']:.3f} L")
    metrics[3].metric(
        "Chance-adjusted volume", f"{totals['chance_constraint_lhs']:.3f} L"
    )
    metrics[4].metric("Modeled fit probability", f"{totals['modeled_fit_probability']:.2%}")

    st.subheader("Selected portfolio")
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

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Capacity and risk")
        st.dataframe(
            pd.DataFrame(
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
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Run diagnostics")
        st.dataframe(
            pd.DataFrame(
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
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("View all estimated item volumes"):
        st.dataframe(
            pd.DataFrame(result["estimated_items"]).rename(
                columns={
                    "name": "Item",
                    "price": "Price",
                    "estimated_volume": "Estimated volume (L)",
                    "selected": "Selected",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    json_download(
        "Download Case 1 results as JSON",
        result,
        "case_1_chance_constrained_result.json",
        "download_case_1",
    )


def render_case_2(result: dict[str, Any]) -> None:
    totals = result["totals"]
    st.success("The sequential arrival scenario is complete.", icon="✅")

    metrics = st.columns(5)
    metrics[0].metric("Total value", money(totals["total_value"]))
    metrics[1].metric("Accepted items", totals["accepted_item_count"])
    metrics[2].metric("Used volume", f"{totals['used_volume']:.3f} L")
    metrics[3].metric("Capacity utilization", f"{totals['capacity_utilization']:.2%}")
    metrics[4].metric("Remaining capacity", f"{totals['remaining_capacity']:.3f} L")

    st.subheader("Arrival decisions")
    decision_frame = pd.DataFrame(result["decisions"]).rename(
        columns={
            "arrival_position": "Arrival",
            "item": "Item",
            "price": "Price",
            "expected_volume": "Expected volume (L)",
            "realized_volume": "Realized volume (L)",
            "bid_price_per_liter": "Bid price / L",
            "approximate_opportunity_cost": "Opportunity cost",
            "approximate_marginal_value": "Marginal value",
            "accepted": "Accepted",
            "remaining_capacity_after": "Capacity after (L)",
            "cumulative_value": "Cumulative value",
            "decision_reason": "Decision reason",
        }
    )
    display_columns = [
        "Arrival",
        "Item",
        "Price",
        "Expected volume (L)",
        "Realized volume (L)",
        "Bid price / L",
        "Opportunity cost",
        "Marginal value",
        "Accepted",
        "Capacity after (L)",
        "Cumulative value",
        "Decision reason",
    ]
    st.dataframe(
        decision_frame[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Expected volume (L)": st.column_config.NumberColumn(format="%.4f"),
            "Realized volume (L)": st.column_config.NumberColumn(format="%.4f"),
            "Bid price / L": st.column_config.NumberColumn(format="%.4f"),
            "Opportunity cost": st.column_config.NumberColumn(format="%.2f"),
            "Marginal value": st.column_config.NumberColumn(format="%.2f"),
            "Capacity after (L)": st.column_config.NumberColumn(format="%.4f"),
            "Cumulative value": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    chart_data = decision_frame.set_index("Arrival")[
        ["Capacity after (L)", "Cumulative value"]
    ]
    st.subheader("Scenario progression")
    st.line_chart(chart_data)

    with st.expander("Accepted items and realized volumes"):
        st.dataframe(
            pd.DataFrame(result["accepted_items"]).rename(
                columns={
                    "name": "Item",
                    "price": "Price",
                    "expected_volume": "Expected volume (L)",
                    "realized_volume": "Realized volume (L)",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    json_download(
        "Download Case 2 scenario as JSON",
        result,
        "case_2_online_scenario.json",
        "download_case_2",
    )


def _comparison_rows(simulation: dict[str, Any]) -> pd.DataFrame:
    baseline = simulation["case_1_baseline"]
    online = simulation["case_2_online"]
    return pd.DataFrame(
        {
            "Statistic": [
                "Average total value",
                "Value standard deviation",
                "95% CI for mean value",
                "Average used volume",
                "Average capacity utilization",
                "Average overflow volume",
                "Probability of fitting",
                "Probability of overflow",
                "Average item count",
            ],
            "Case 1: fixed stochastic portfolio": [
                money(baseline["average_total_value"]),
                f"{baseline['value_standard_deviation']:.2f}",
                (
                    f"[{baseline['mean_value_95_percent_ci'][0]:.2f}, "
                    f"{baseline['mean_value_95_percent_ci'][1]:.2f}]"
                ),
                f"{baseline['average_used_volume']:.3f} L",
                f"{baseline['average_capacity_utilization']:.2%}",
                f"{baseline['average_overflow_volume']:.3f} L",
                f"{baseline['fit_probability']:.2%}",
                f"{baseline['overflow_probability']:.2%}",
                f"{baseline['average_item_count']:.2f}",
            ],
            "Case 2: online bid-price policy": [
                money(online["average_total_value"]),
                f"{online['value_standard_deviation']:.2f}",
                (
                    f"[{online['mean_value_95_percent_ci'][0]:.2f}, "
                    f"{online['mean_value_95_percent_ci'][1]:.2f}]"
                ),
                f"{online['average_used_volume']:.3f} L",
                f"{online['average_capacity_utilization']:.2%}",
                f"{online['average_overflow_volume']:.3f} L",
                f"{online['fit_probability']:.2%}",
                f"{online['overflow_probability']:.2%}",
                f"{online['average_item_count']:.2f}",
            ],
        }
    )


def render_case_3(bundle: dict[str, Any]) -> None:
    simulation = bundle["simulation"]
    baseline = simulation["case_1_baseline"]
    online = simulation["case_2_online"]
    comparison = simulation["comparison"]
    st.success(
        f"Completed {simulation['parameters']['runs']:,} paired simulation runs.",
        icon="✅",
    )

    case_1_column, case_2_column = st.columns(2, gap="large")
    with case_1_column:
        st.subheader("Case 1 — fixed portfolio")
        row = st.columns(2)
        row[0].metric("Average value", money(baseline["average_total_value"]))
        row[1].metric("Average utilization", f"{baseline['average_capacity_utilization']:.2%}")
        row = st.columns(2)
        row[0].metric("Average overflow", f"{baseline['average_overflow_volume']:.3f} L")
        row[1].metric("Fit probability", f"{baseline['fit_probability']:.2%}")
    with case_2_column:
        st.subheader("Case 2 — online policy")
        row = st.columns(2)
        row[0].metric(
            "Average value",
            money(online["average_total_value"]),
            delta=f"{comparison['average_value_difference_case_2_minus_case_1']:+.2f}",
        )
        row[1].metric(
            "Average utilization", f"{online['average_capacity_utilization']:.2%}"
        )
        row = st.columns(2)
        row[0].metric(
            "Average overflow",
            f"{online['average_overflow_volume']:.3f} L",
            delta=f"{comparison['average_overflow_reduction']:.3f} L reduction",
        )
        row[1].metric(
            "Fit probability",
            f"{online['fit_probability']:.2%}",
            delta=f"{comparison['fit_probability_improvement']:+.2%}",
        )

    st.subheader("Side-by-side comparative statistics")
    st.dataframe(_comparison_rows(simulation), hide_index=True, width="stretch")

    scenario_frame = pd.DataFrame(simulation["scenario_results"])
    st.subheader("Value across paired runs")
    st.line_chart(
        scenario_frame.set_index("run")[
            ["case_1_total_value", "case_2_total_value"]
        ].rename(
            columns={
                "case_1_total_value": "Case 1",
                "case_2_total_value": "Case 2",
            }
        )
    )

    with st.expander("View run-level simulation results"):
        st.dataframe(scenario_frame, hide_index=True, width="stretch")

    json_download(
        "Download Case 3 comparison as JSON",
        bundle,
        "case_3_policy_comparison.json",
        "download_case_3",
    )


st.subheader("Input data")
upload_left, upload_right = st.columns(2, gap="large")
with upload_left:
    items_file = st.file_uploader(
        "Items data",
        type=["json"],
        key="items_file",
        help="JSON array with name and price for every item.",
    )
    st.caption('Expected fields: `[{"name": "A1", "price": 98}, ...]`')
with upload_right:
    packages_file = st.file_uploader(
        "Package history",
        type=["json"],
        key="packages_file",
        help="JSON array with measured total_volume and the items in each package.",
    )
    st.caption(
        'Expected fields: `[{"total_volume": 36.04, "items": ["A3", "A28"]}, ...]`'
    )

if items_file is not None and packages_file is not None:
    input_fingerprint = hashlib.sha256(
        items_file.getvalue() + b"\0" + packages_file.getvalue()
    ).hexdigest()
    if st.session_state.get("_input_fingerprint") != input_fingerprint:
        for result_key in ("case_1_result", "case_2_result", "case_3_result"):
            st.session_state.pop(result_key, None)
        st.session_state["_input_fingerprint"] = input_fingerprint
    st.caption("Both input files are ready. Choose a case below.")


case_1_tab, case_2_tab, case_3_tab = st.tabs(
    [
        "Case 1 — Baseline portfolio",
        "Case 2 — Sequential arrivals",
        "Case 3 — Policy simulation",
    ]
)


with case_1_tab:
    st.markdown(
        """
        **Baseline stochastic-programming case.** Item volumes are estimated from
        package history, and a fixed binary portfolio is selected before any exact
        volumes are observed. A Gaussian chance constraint adds a confidence buffer
        to protect the 40-liter capacity.
        """
    )
    settings = st.columns(4)
    case_1_capacity = settings[0].number_input(
        "Capacity (L)", min_value=0.01, value=40.0, step=1.0, key="case_1_capacity"
    )
    case_1_confidence = settings[1].slider(
        "Minimum fit confidence",
        min_value=50.1,
        max_value=99.9,
        value=95.0,
        step=0.1,
        key="case_1_confidence",
    )
    case_1_variance = settings[2].number_input(
        "Measurement-error variance",
        min_value=0.0001,
        value=2.0,
        step=0.1,
        key="case_1_variance",
    )
    case_1_time_limit = settings[3].number_input(
        "Time limit (seconds)",
        min_value=5,
        max_value=600,
        value=60,
        step=5,
        key="case_1_time_limit",
    )
    run_case_1 = st.button(
        "Solve Case 1",
        type="primary",
        icon="▶",
        width="stretch",
        key="run_case_1",
    )
    if run_case_1:
        st.session_state.pop("case_1_result", None)
        try:
            with st.spinner("Estimating volumes and solving the chance-constrained model…"):
                problem, regression = load_model(items_file, packages_file)
                st.session_state["case_1_result"] = solve_chance_constrained(
                    problem,
                    regression,
                    capacity=float(case_1_capacity),
                    confidence=float(case_1_confidence / 100.0),
                    noise_variance=float(case_1_variance),
                    time_limit=float(case_1_time_limit),
                )
        except (InputDataError, OptimizationError) as exc:
            st.error(str(exc), icon="⚠️")
        except Exception as exc:
            st.exception(exc)

    st.divider()
    st.subheader("Case 1 results")
    if "case_1_result" in st.session_state:
        render_case_1(st.session_state["case_1_result"])
    else:
        st.info("Upload both files and select **Solve Case 1**.")


with case_2_tab:
    st.markdown(
        """
        **One sequential scenario.** Every item arrives once in random order. Its
        exact volume is revealed on arrival, and the decision is irreversible.
        Before each decision, a deterministic fractional-knapsack LP estimates the
        marginal value of remaining capacity. The item is accepted when it fits and
        its value exceeds the bid-price opportunity cost.
        """
    )
    settings = st.columns(3)
    case_2_capacity = settings[0].number_input(
        "Capacity (L)", min_value=0.01, value=40.0, step=1.0, key="case_2_capacity"
    )
    case_2_variance = settings[1].number_input(
        "Volume-realization variance",
        min_value=0.0001,
        value=2.0,
        step=0.1,
        key="case_2_variance",
    )
    case_2_seed = settings[2].number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=2026,
        step=1,
        key="case_2_seed",
    )
    run_case_2 = st.button(
        "Realize arrivals and solve Case 2",
        type="primary",
        icon="▶",
        width="stretch",
        key="run_case_2",
    )
    if run_case_2:
        st.session_state.pop("case_2_result", None)
        try:
            with st.spinner("Realizing volumes and applying the online bid-price policy…"):
                problem, regression = load_model(items_file, packages_file)
                st.session_state["case_2_result"] = solve_online_bid_price_scenario(
                    problem,
                    regression,
                    capacity=float(case_2_capacity),
                    noise_variance=float(case_2_variance),
                    seed=int(case_2_seed),
                )
        except (InputDataError, OptimizationError) as exc:
            st.error(str(exc), icon="⚠️")
        except Exception as exc:
            st.exception(exc)

    st.divider()
    st.subheader("Case 2 results")
    if "case_2_result" in st.session_state:
        render_case_2(st.session_state["case_2_result"])
    else:
        st.info("Upload both files and select **Realize arrivals and solve Case 2**.")


with case_3_tab:
    st.markdown(
        """
        **Paired Monte Carlo simulation.** Each run generates one joint volume
        realization. The fixed Case 1 portfolio and the Case 2 online policy are
        evaluated on that same realization. Case 1 may overflow; Case 2 rejects any
        item that does not fit. Common random numbers make the comparison more
        precise.
        """
    )
    first_row = st.columns(3)
    case_3_capacity = first_row[0].number_input(
        "Capacity (L)", min_value=0.01, value=40.0, step=1.0, key="case_3_capacity"
    )
    case_3_confidence = first_row[1].slider(
        "Case 1 fit confidence",
        min_value=50.1,
        max_value=99.9,
        value=95.0,
        step=0.1,
        key="case_3_confidence",
    )
    case_3_variance = first_row[2].number_input(
        "Volume-realization variance",
        min_value=0.0001,
        value=2.0,
        step=0.1,
        key="case_3_variance",
    )
    second_row = st.columns(3)
    case_3_runs = second_row[0].number_input(
        "Simulation runs",
        min_value=10,
        max_value=10_000,
        value=500,
        step=100,
        key="case_3_runs",
    )
    case_3_seed = second_row[1].number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=2026,
        step=1,
        key="case_3_seed",
    )
    case_3_time_limit = second_row[2].number_input(
        "Case 1 time limit (seconds)",
        min_value=5,
        max_value=600,
        value=60,
        step=5,
        key="case_3_time_limit",
    )
    run_case_3 = st.button(
        "Run paired simulation",
        type="primary",
        icon="▶",
        width="stretch",
        key="run_case_3",
    )
    if run_case_3:
        st.session_state.pop("case_3_result", None)
        try:
            with st.spinner("Solving Case 1 and running paired policy simulations…"):
                problem, regression = load_model(items_file, packages_file)
                baseline_result = solve_chance_constrained(
                    problem,
                    regression,
                    capacity=float(case_3_capacity),
                    confidence=float(case_3_confidence / 100.0),
                    noise_variance=float(case_3_variance),
                    time_limit=float(case_3_time_limit),
                )
                simulation = simulate_baseline_and_online(
                    problem,
                    regression,
                    baseline_result,
                    capacity=float(case_3_capacity),
                    noise_variance=float(case_3_variance),
                    runs=int(case_3_runs),
                    seed=int(case_3_seed),
                )
                st.session_state["case_3_result"] = {
                    "baseline_solution": baseline_result,
                    "simulation": simulation,
                }
        except (InputDataError, OptimizationError) as exc:
            st.error(str(exc), icon="⚠️")
        except Exception as exc:
            st.exception(exc)

    st.divider()
    st.subheader("Case 3 results")
    if "case_3_result" in st.session_state:
        render_case_3(st.session_state["case_3_result"])
    else:
        st.info("Upload both files and select **Run paired simulation**.")


with st.expander("Modeling assumptions"):
    st.markdown(
        """
        - Volume realizations are sampled jointly from
          `Normal(estimated volumes, noise variance × (AᵀA)⁻¹)` and resampled if
          any physical volume is non-positive.
        - Each item arrives exactly once in a uniformly random order.
        - The online policy uses expected volumes for future items and the exact
          realized volume for the item currently being considered.
        - Case 1 retains its fixed portfolio in every simulation run. Its value is
          reported even when realized total volume exceeds capacity; overflow and
          feasibility are reported separately.
        - Case 2 never intentionally overflows because an item is rejected whenever
          its realized volume exceeds the remaining capacity.
        """
    )
