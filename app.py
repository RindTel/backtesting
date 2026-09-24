"""Streamlit UI for the backtester: backtest, leak test, runs and sweep.

    streamlit run app.py

Pages live in ``ui.py``, styles in ``ui.css`` and ``.streamlit/config.toml``.
Every run made here is also logged to MLflow.
"""

import streamlit as st

import ui

st.set_page_config(page_title="Backtester", page_icon=":material/monitoring:", layout="wide")
ui.inject_css()

nav = st.navigation(
    [
        st.Page(ui.page_backtest, title="Backtest", icon=":material/show_chart:", url_path="backtest", default=True),
        st.Page(ui.page_leak, title="Leak test", icon=":material/compare_arrows:", url_path="leak"),
        st.Page(ui.page_runs, title="Runs", icon=":material/history:", url_path="runs"),
        st.Page(ui.page_sweep, title="Sweep", icon=":material/grid_on:", url_path="sweep"),
    ],
    position="top",
)
nav.run()
