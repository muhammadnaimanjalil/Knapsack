"""Small visual system for a calm, readable Streamlit interface."""

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          .block-container {max-width: 1120px; padding-top: 1.6rem; padding-bottom: 4rem;}
          .app-hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.20);
            border-radius: 16px; margin-bottom: 1.1rem;
            background: linear-gradient(135deg, rgba(38,99,235,.10), rgba(255,255,255,.02));}
          .app-hero h1 {font-size: 2rem; margin: 0; letter-spacing: -.03em;}
          .app-hero p {margin: .4rem 0 0; opacity: .78;}
          div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.18);
            padding: .75rem .9rem; border-radius: 12px;}
          div[data-testid="stFileUploader"] {border-radius: 12px;}
          div[data-baseweb="tab-list"] {gap: .35rem;}
          button[data-baseweb="tab"] {padding-left: .8rem; padding-right: .8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
