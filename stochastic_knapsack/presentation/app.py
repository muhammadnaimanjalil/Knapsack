"""Composition root for the Streamlit application.

This module arranges presentation components only. Domain calculations are
accessed through the three application services.
"""

import logging

import streamlit as st

from stochastic_knapsack.presentation.case_1_tab import render_case_1_tab
from stochastic_knapsack.presentation.case_2_tab import render_case_2_tab
from stochastic_knapsack.presentation.case_3_tab import render_case_3_tab
from stochastic_knapsack.presentation.input_panel import render_input_panel
from stochastic_knapsack.presentation.theme import apply_theme


def run_app() -> None:
    """Configure and render the complete interactive application."""
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(
        page_title="Stochastic Knapsack: Offline and Online Configurations",
        page_icon="🎒", layout="wide", initial_sidebar_state="collapsed"
    )
    apply_theme()
    st.markdown(
        """
        <div class="app-hero">
          <h1>Stochastic Knapsack</h1>
          <p>Compare a portfolio selected before volumes are known with an adaptive policy that decides as items arrive.</p>
        </div>
        """, unsafe_allow_html=True
    )
    with st.expander("About the models"):
        st.markdown(
            """The application estimates missing item volumes from historical
            package measurements. The offline case uses a Gaussian chance constraint;
            the online case uses a dynamic bid price; and the comparison evaluates
            both policies with paired Monte Carlo scenarios."""
        )
    render_input_panel()
    offline_tab, online_tab, comparison_tab = st.tabs(
        ["Offline portfolio", "Online decisions", "Compare policies"]
    )
    with offline_tab:
        st.caption("Case 1 · Select everything before exact volumes are known")
        render_case_1_tab()
    with online_tab:
        st.caption("Case 2 · Decide after each item's volume is revealed")
        render_case_2_tab()
    with comparison_tab:
        st.caption("Case 3 · Compare long-run policy performance")
        render_case_3_tab()
