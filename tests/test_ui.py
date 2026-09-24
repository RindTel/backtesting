"""The Streamlit UI, driven headlessly with Streamlit's AppTest.

Pages run on the committed sample, and MLflow logging goes to a throwaway store.
"""

from functools import partial
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import scripts.run_backtest as rb
from mlflow_logger import log_backtest_run
from strategies import STRATEGIES

TIMEOUT = 180
STRATEGY_LABELS = {name: cls.label for name, cls in STRATEGIES.items()}


@pytest.fixture(autouse=True)
def scratch_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    logger = partial(log_backtest_run, tracking_uri=uri, artifact_root=(tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(rb, "log_backtest_run", logger)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    st.cache_data.clear()  # the UI caches backtests; each test starts from a clean store


def _page(name: str) -> None:
    import ui

    getattr(ui, name)()


def page(name: str) -> AppTest:
    return AppTest.from_function(_page, args=(name,), default_timeout=TIMEOUT).run()


def texts(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_entry_point_loads() -> None:
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Backtest"]


def test_backtest_page_is_result_first_with_the_documented_numbers() -> None:
    at = page("page_backtest")  # no click: the defaults run on first visit
    assert not at.exception
    shown = {m.label: m.value for m in at.metric}
    # Same numbers as README / demo_script.md for the 10/30 crossover on the sample.
    assert shown["Total return"] == "+598.1%"
    assert shown["Sharpe"] == "1.09"
    assert shown["Fills"] == "68"
    assert "look-ahead bias" not in texts(at)


def test_backtest_page_flags_the_broken_engine() -> None:
    at = page("page_backtest")
    at.radio(key="engine").set_value("Broken").run()
    at.button(key="bt_run").click().run()
    assert not at.exception
    assert "look-ahead bias" in texts(at)
    assert {m.label: m.value for m in at.metric}["Total return"] == "+1285.7%"


def test_leak_page_faces_off_both_engines_on_the_same_data() -> None:
    at = page("page_leak")
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table.columns) == ["Metric", "Correct engine", "Broken engine", "Buy & hold"]
    assert table.set_index("Metric").loc["Total return"].tolist() == ["+598.1%", "+1285.7%", "+1197.6%"]
    assert "Same data" in texts(at) and "a58795bacb31c078" in texts(at)


def test_runs_page_lists_logged_runs() -> None:
    assert "No runs logged yet" in texts(page("page_runs"))

    page("page_backtest")  # its first-visit run is logged
    at = page("page_runs")
    assert not at.exception
    assert at.dataframe[0].value.shape[0] == 1


def test_sweep_page_picks_in_sample_and_reports_out_of_sample() -> None:
    at = page("page_sweep")
    at.text_input(key="sw_short").set_value("5").run()
    at.text_input(key="sw_long").set_value("30,50").run()
    at.button(key="sw_run").click().run()
    assert not at.exception
    assert "Picked by in-sample Sharpe" in texts(at)
    assert at.dataframe[0].value.shape[0] == 2


def test_sweep_page_rejects_bad_windows() -> None:
    at = page("page_sweep")
    at.text_input(key="sw_short").set_value("five").run()
    at.button(key="sw_run").click().run()
    assert any("comma-separated whole numbers" in e.value for e in at.error)


def test_missing_data_shows_fetch_command() -> None:
    at = page("page_backtest")
    at.text_input(key="set_symbol").set_value("NOPEUSDT").run()
    assert any("No stored data for NOPEUSDT" in e.value for e in at.error)
    assert any("scripts.fetch_data" in c.value for c in at.code)


def test_intro_is_open_on_first_visit_and_hints_explain_the_numbers() -> None:
    at = page("page_backtest")
    intro = at.expander[0]
    assert intro.label == "What am I looking at?"
    assert "buy & hold" in " ".join(m.value for m in intro.markdown)
    hints = " ".join(c.value for c in at.caption)
    assert r"\$10,000 became \$69,810" in hints  # escaped, or Markdown renders it as a LaTeX formula
    assert "Return per unit of risk" in hints


@pytest.mark.parametrize("strategy", ["momentum", "breakout", "rsi_reversion", "buy_and_hold"])
def test_every_strategy_runs_from_the_ui(strategy: str) -> None:
    at = page("page_backtest")
    at.selectbox(key="bt_strategy").set_value(strategy).run()
    at.button(key="bt_run").click().run()
    assert not at.exception
    assert {m.label for m in at.metric} >= {"Total return", "Sharpe", "Max drawdown"}
    assert STRATEGY_LABELS[strategy] in texts(at)


def test_leak_page_works_for_another_strategy() -> None:
    at = page("page_leak")
    at.selectbox(key="leak_strategy").set_value("breakout").run()
    at.button(key="leak_run").click().run()
    assert not at.exception
    assert list(at.dataframe[0].value.columns) == ["Metric", "Correct engine", "Broken engine", "Buy & hold"]


@pytest.mark.parametrize("bad", ["../../ETC", "<IMG SRC=X>"])
def test_ui_rejects_unsafe_symbols_cleanly(bad: str) -> None:
    at = page("page_backtest")
    at.text_input(key="set_symbol").set_value(bad).run()
    assert not at.exception
    assert any("invalid symbol" in e.value for e in at.error)


def test_data_version_changes_when_stored_data_changes(tmp_path) -> None:
    """Review finding: the UI cache was keyed on settings only, so re-fetched data
    returned stale results. The key now includes each partition's mtime and size."""
    import os

    import ui
    from data_loader import write_partitioned
    from tests.helpers import make_bars

    (written,) = write_partitioned(make_bars([100.0, 101.0], symbol="TEST"), "binance", root=tmp_path)
    before = ui.data_version(str(tmp_path), "TEST")
    os.utime(written, ns=(1, 1))
    assert ui.data_version(str(tmp_path), "TEST") != before


def _contrast(a: str, b: str) -> float:
    def lum(h: str) -> float:
        c = [int(h.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_theme_primary_colour_keeps_streamlits_white_text_readable() -> None:
    """Regression: primaryColor was off-white, and Streamlit hard-codes WHITE text on the
    primary colour (multiselect tags, date-picker selection), so the Runs page's
    Engine tags rendered as blank white boxes (contrast 1.17:1)."""
    import tomllib

    theme = tomllib.loads((Path(__file__).parent.parent / ".streamlit" / "config.toml").read_text())["theme"]
    assert _contrast("#ffffff", theme["primaryColor"]) >= 4.5  # text drawn on it
    assert _contrast(theme["primaryColor"], theme["backgroundColor"]) >= 3.0  # control visible on the page
