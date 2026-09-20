"""Centralized Streamlit session-state keys and invalidation rules."""

from __future__ import annotations

import streamlit as st


RESULT_KEYS = ("offline_result", "online_result", "simulation_result")


def clear_results() -> None:
    """Discard results when the underlying data source changes."""
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


def store_prepared_data(fingerprint: str, problem: object, volume_model: object) -> None:
    """Store validated shared inputs and invalidate results only when necessary."""
    if st.session_state.get("data_fingerprint") != fingerprint:
        clear_results()
    st.session_state["data_fingerprint"] = fingerprint
    st.session_state["problem"] = problem
    st.session_state["volume_model"] = volume_model


def prepared_data() -> tuple[object, object] | None:
    problem = st.session_state.get("problem")
    volume_model = st.session_state.get("volume_model")
    return (problem, volume_model) if problem is not None and volume_model is not None else None
