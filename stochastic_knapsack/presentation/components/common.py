"""Formatting, download, and status helpers for Streamlit views."""

from __future__ import annotations

from typing import Any

import streamlit as st

from stochastic_knapsack.presentation.serialization import result_json


def value_text(value: float) -> str:
    return f"{value:,.2f}"


def liters(value: float) -> str:
    return f"{value:,.2f} L"


def percent(value: float) -> str:
    return f"{value:.1%}"


def download_result(result: Any, file_name: str, key: str) -> None:
    st.download_button(
        "Download full result (JSON)",
        data=result_json(result),
        file_name=file_name,
        mime="application/json",
        key=key,
        width="stretch",
    )


def require_data() -> bool:
    """Show one consistent prompt when a tab is used before data preparation."""
    if "problem" in st.session_state and "volume_model" in st.session_state:
        return True
    st.info("Complete **Step 1 — Choose data** above before running this case.")
    return False
