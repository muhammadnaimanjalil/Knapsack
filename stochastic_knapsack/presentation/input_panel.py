"""Shared data-source selection and validation panel."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from stochastic_knapsack.application.common import prepare_problem
from stochastic_knapsack.data.json_loader import load_json_bytes, load_problem_paths
from stochastic_knapsack.domain.exceptions import InputDataError
from stochastic_knapsack.presentation.state import store_prepared_data


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "tests" / "Test Scenario"


def _prepare(items_payload: object, packages_payload: object, fingerprint: str) -> None:
    problem, volume_model = prepare_problem(items_payload, packages_payload)
    store_prepared_data(fingerprint, problem, volume_model)


def render_input_panel() -> None:
    """Render one compact entry point for example data or uploaded JSON files."""
    with st.container(border=True):
        st.subheader("Step 1 — Choose data")
        source = st.radio(
            "Data source",
            ("Example scenario", "Upload files"),
            horizontal=True,
            label_visibility="collapsed",
        )
        try:
            if source == "Example scenario":
                st.caption("Use the repository's 60-item, 1,000-package scenario.")
                if st.button("Load example data", type="primary", width="stretch"):
                    items, packages = load_problem_paths(
                        EXAMPLE_DIRECTORY / "items.json",
                        EXAMPLE_DIRECTORY / "packages.json",
                    )
                    _prepare(items, packages, "repository-example-v1")
            else:
                left, right = st.columns(2)
                items_file = left.file_uploader(
                    "Items data",
                    type=["json"],
                    help="JSON array containing item names and values in the price field.",
                )
                packages_file = right.file_uploader(
                    "Package history",
                    type=["json"],
                    help="JSON array containing package item lists and measured total volumes.",
                )
                ready = items_file is not None and packages_file is not None
                if st.button(
                    "Validate uploaded data",
                    type="primary",
                    width="stretch",
                    disabled=not ready,
                ):
                    items_bytes = items_file.getvalue()
                    packages_bytes = packages_file.getvalue()
                    fingerprint = hashlib.sha256(items_bytes + packages_bytes).hexdigest()
                    _prepare(
                        load_json_bytes(items_bytes, "items.json"),
                        load_json_bytes(packages_bytes, "packages.json"),
                        fingerprint,
                    )
        except InputDataError as exc:
            st.error(str(exc), icon="⚠️")

        if "problem" in st.session_state:
            problem = st.session_state["problem"]
            model = st.session_state["volume_model"]
            st.success(
                f"Data ready: {len(problem.items):,} items and "
                f"{len(problem.packages):,} package observations."
            )
            with st.expander("Data diagnostics"):
                st.write(
                    {
                        "Regression rank": model.rank,
                        "Condition number": round(model.condition_number, 4),
                        "Regression RMSE": round(model.rmse, 6),
                        "Residual variance": round(model.residual_variance, 6),
                    }
                )
