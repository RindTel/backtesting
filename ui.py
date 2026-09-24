"""The Streamlit UI's pages and helpers. ``app.py`` wires them into navigation.

A thin layer over the same functions the CLI uses (``scripts.run_backtest.backtest``,
``scripts.sweep.run_sweep``), so every run made here is also logged to MLflow,
and no engine logic lives in the UI.

Design system (DESIGN.md): near-black workspace, white/grey type, Geist + Geist Mono,
one signal red that only ever means "broken engine", "loss" or "drawdown".
Layout grammar on every page: slim control bar, KPI strip, hero chart, detail row.
"""

import html
import inspect
import math
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st
from mlflow.tracking import MlflowClient

import config
from data_loader import load, partition_files, validate_symbol
from scripts.run_backtest import backtest, parse_args
from scripts.sweep import run_sweep
from strategies import STRATEGIES

SAMPLE_ROOT = str(config.PROJECT_ROOT / "data" / "sample")
CSS = (Path(__file__).parent / "ui.css").read_text()

# Tokens (mirror ui.css / .streamlit/config.toml).
INK, INK_2, INK_3, LINE, GRID, SIGNAL = "#ededed", "#a1a1a1", "#6e6e6e", "#222222", "#1c1c1c", "#f0544f"
SOLID, DASHED = [1, 0], [4, 4]
# Past-runs overlays: grey tones + line style tell runs apart; broken runs keep the signal colour.
RUN_GREYS = ["#ededed", "#bababa", "#888888", "#6e6e6e", "#565656"]
RUN_DASHES = [[1, 0], [6, 3], [2, 3], [8, 3, 2, 3], [1, 2]]
MAX_OVERLAY = len(RUN_GREYS)


@alt.theme.register("backtest_dark", enable=True)
def _chart_theme() -> alt.theme.ThemeConfig:
    return {
        "config": {
            "background": "transparent",
            "font": "Geist",
            "view": {"stroke": None},
            "axis": {
                "labelColor": INK_2,
                "labelFont": "Geist Mono",
                "labelFontSize": 11,
                "titleColor": INK_2,
                "titleFontWeight": 500,
                "gridColor": GRID,
                "domain": False,
                "tickColor": LINE,
                "tickSize": 4,
            },
            "axisX": {"grid": False},
            "legend": {
                "labelColor": INK,
                "labelFont": "Geist",
                "labelFontSize": 12,
                "orient": "top-left",
                "title": None,
                "symbolStrokeWidth": 2,
                "symbolType": "stroke",
            },
            "title": {"color": INK, "font": "Geist", "fontSize": 13, "fontWeight": 600, "anchor": "start"},
            "text": {"font": "Geist Mono"},
        }
    }


def inject_css() -> None:
    st.html(f"<style>{CSS}</style>")


# --- Data & costs (shared by every page, lives in a popover) -----------------


def data_version(root: str, symbol: str) -> tuple:
    """(path, mtime, size) of each stored partition: changes whenever the data on disk changes.

    Part of every cache key below, so re-fetched data can never be answered from a stale cache.
    """
    return tuple(
        (str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in partition_files(symbol, "binance", Path(root))
    )


@st.cache_data(show_spinner=False)
def data_range(root: str, symbol: str, version: tuple) -> tuple[date, date] | None:
    try:
        ds = load(symbol, "binance", root=Path(root))
    except FileNotFoundError:
        return None
    return ds.start, ds.end


def data_and_costs(bar) -> list[str]:
    """The "Data & costs" popover in a control bar, returned as ``run_backtest`` CLI flags.

    Widget keys are shared across pages, so a setting chosen on one page holds on the others.
    """
    with bar.popover("Data & costs", icon=":material/tune:"):
        source = st.radio(
            "Data", ["Committed sample", "Downloaded"], key="set_source", horizontal=True,
            help="Committed sample: Binance Vision BTCUSDT 2020-2024 (CC BY-NC-SA 4.0). Downloaded: data/processed/.",
        )  # fmt: skip
        root = SAMPLE_ROOT if source == "Committed sample" else str(config.PROCESSED_DATA_DIR)
        symbol = st.text_input("Binance spot pair", "BTCUSDT", key="set_symbol").strip().upper()
        try:
            validate_symbol(symbol)
            invalid = None
        except ValueError as e:
            invalid = str(e)
        available = None if invalid else data_range(root, symbol, data_version(root, symbol))
        if available:
            picked = st.date_input(
                "Date range", value=available, min_value=available[0], max_value=available[1], key="set_range"
            )
        capital = st.number_input(
            "Starting capital ($)",
            min_value=100.0,
            value=config.DEFAULT_INITIAL_CAPITAL,
            step=1000.0,
            key="set_capital",
        )
        c1, c2 = st.columns(2)
        fee = c1.number_input("Fee (bps)", min_value=0.1, value=config.DEFAULT_FEE_RATE * 1e4, key="set_fee")
        spread = c2.number_input(
            "Spread (bps)", min_value=0.1, value=config.DEFAULT_SLIPPAGE_RATE * 1e4, key="set_spread"
        )
        impact = c1.number_input(
            "Impact k", min_value=0.0, value=config.DEFAULT_IMPACT_COEFFICIENT, step=0.1, key="set_impact",
            help="Square-root market impact: k · σ · √(order $ / ADV). 0 turns it off.",
        )  # fmt: skip
        cap = c2.number_input(
            "Volume cap (%)", 1, 100, int(config.DEFAULT_MAX_PARTICIPATION * 100), key="set_cap",
            help="Largest fill per day as a share of average daily dollar volume. The rest carries over.",
        )  # fmt: skip

    if invalid:
        st.error(invalid)
        st.stop()
    if available is None:
        st.error(f"No stored data for {symbol}. Fetch it first:")
        st.code(f"python -m scripts.fetch_data --source binance --symbol {symbol} --start 2020-01-01 --end 2024-12-31")
        st.stop()
    if len(picked) != 2:
        st.info("Pick an end date in Data & costs.")
        st.stop()
    return [
        "--source", "binance", "--symbol", symbol, "--start", str(picked[0]), "--end", str(picked[1]),
        "--data-root", root, "--capital", str(capital), "--fee-rate", str(fee / 1e4),
        "--slippage-rate", str(spread / 1e4), "--impact-coefficient", str(impact),
        "--max-participation", str(cap / 100),
    ]  # fmt: skip


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_backtest(argv: tuple[str, ...], version: tuple):
    # Identical settings on identical data return the stored result instead of
    # logging a duplicate MLflow run. `version` is only part of the cache key.
    return backtest(parse_args(list(argv)))


def run(argv: list[str]):
    """A (cached) backtest, with errors shown inline instead of a traceback."""
    try:
        a = parse_args(argv)
        return _cached_backtest(tuple(argv), data_version(str(a.data_root), a.symbol))
    except (ValueError, TypeError, FileNotFoundError) as e:
        st.error(str(e))
        st.stop()


def control_bar(key: str):
    return st.container(horizontal=True, vertical_alignment="bottom", gap="small", key=f"bar-{key}")


# --- Formatting ----------------------------------------------------------------


def pct(x: float | None, signed: bool = True) -> str:
    return "n/a" if x is None or math.isnan(x) else (f"{x:+.1%}" if signed else f"{x:.1%}")


def num(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.2f}"


def is_negative(x: float | None) -> bool:
    return x is not None and not math.isnan(x) and x < 0


def note(html: str) -> None:
    st.markdown(f'<p class="note">{html}</p>', unsafe_allow_html=True)


def kpi_strip(m, page: str, benchmark=None, equity: pl.Series | None = None) -> None:
    """Six headline numbers separated by hairlines, each with a plain-English hint underneath.

    Negative values render in the signal colour.
    """
    delta = f"{(m.total_return - benchmark.total_return) * 100:+.1f} pts vs buy & hold" if benchmark else None
    # Escaped: captions are Markdown, and two "$" in one line would render as a LaTeX formula.
    grew = f"\\${equity[0]:,.0f} became \\${equity[-1]:,.0f}" if equity is not None else "Growth over the whole period"
    items = [
        ("Total return", pct(m.total_return), delta, is_negative(m.total_return), grew),
        ("CAGR", pct(m.cagr), None, is_negative(m.cagr), "The same growth, as a yearly rate"),
        ("Sharpe", num(m.sharpe_ratio), None, is_negative(m.sharpe_ratio), "Return per unit of risk. Above 1 is good"),
        ("Max drawdown", pct(m.max_drawdown.depth), None, True, "Worst drop from a previous high"),
        (
            "Win rate",
            pct(m.win_rate, signed=False),
            f"{m.num_round_trips} closed trips",
            False,
            "Share of closed trades that made money",
        ),
        ("Fills", str(m.num_trades), None, False, "Buy and sell orders executed"),
    ]
    for i, (col, (label, value, sub, negative, hint)) in enumerate(zip(st.columns(6, gap="small"), items, strict=True)):
        with col, st.container(key=f"{'neg' if negative else 'kpi'}-{page}-{i}"):
            st.metric(label, value, sub, delta_color="off" if label == "Win rate" else "normal")
            st.caption(hint)


# --- Charts --------------------------------------------------------------------


def equity_chart(
    curves: dict[str, pl.DataFrame], styles: dict[str, tuple[str, list[int]]], log_scale: bool, height: int = 460
) -> alt.LayerChart:
    """Equity curves on one shared axis. Series differ by colour AND line style; hover shows a crosshair."""
    data = pl.concat(
        c.select("timestamp", "equity").with_columns(pl.lit(name).alias("series")) for name, c in curves.items()
    )
    names = list(curves)
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=[styles[n][0] for n in names]),
        legend=alt.Legend() if len(names) > 1 else None,
    )
    dash = alt.StrokeDash("series:N", scale=alt.Scale(domain=names, range=[styles[n][1] for n in names]), legend=None)
    x = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%Y", labelAngle=0, tickCount="year"))
    y = alt.Y(
        "equity:Q",
        title=None,
        scale=alt.Scale(type="log" if log_scale else "linear", zero=False),
        axis=alt.Axis(format="$~s", tickCount=5),
    )
    hover = alt.selection_point(fields=["timestamp"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(data).encode(x=x, y=y, color=color)
    lines = base.mark_line(strokeWidth=2).encode(strokeDash=dash)
    points = base.mark_point(size=50, filled=True, stroke="#0a0a0a", strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("series:N", title="Run"),
            alt.Tooltip("timestamp:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("equity:Q", title="Equity", format="$,.0f"),
        ],
    )
    rule = (
        alt.Chart(data)
        .mark_rule(color=INK_3, strokeWidth=1)
        .encode(x=x, opacity=alt.condition(hover, alt.value(1), alt.value(0)))
        .add_params(hover)
    )
    return (lines + rule + points).properties(height=height)


def drawdown_chart(equity_curve: pl.DataFrame, height: int = 220) -> alt.LayerChart:
    """Underwater curve: how far below its running peak the account is. The signal colour means loss."""
    data = equity_curve.select("timestamp", (pl.col("equity") / pl.col("equity").cum_max() - 1).alias("drawdown"))
    base = alt.Chart(data).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%Y", labelAngle=0, tickCount="year")),
        y=alt.Y("drawdown:Q", title=None, axis=alt.Axis(format="%", tickCount=4)),
    )
    area = base.mark_area(color=SIGNAL, opacity=0.18, line=False)
    line = base.mark_line(color=SIGNAL, strokeWidth=1.5).encode(
        tooltip=[
            alt.Tooltip("timestamp:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("drawdown:Q", title="Below peak", format=".1%"),
        ]
    )
    return (area + line).properties(height=height)


def sharpe_heatmap(rows: pl.DataFrame, field: str, title: str, domain: list[float], pick: dict) -> alt.LayerChart:
    base = alt.Chart(rows, title=title).encode(
        x=alt.X("long:O", title="Long window", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("short:O", title="Short window", sort="descending"),
    )
    cells = base.mark_rect(cornerRadius=6, stroke="#0a0a0a", strokeWidth=3).encode(
        color=alt.Color(f"{field}:Q", scale=alt.Scale(domain=domain, range=["#1a1a1a", INK]), legend=None),
        tooltip=["short", "long", alt.Tooltip(f"{field}:Q", format=".3f", title="Sharpe")],
    )
    labels = base.mark_text(fontSize=14).encode(
        text=alt.Text(f"{field}:Q", format=".2f"),
        color=alt.condition(alt.datum[field] > sum(domain) / 2, alt.value("#0a0a0a"), alt.value(INK)),
    )
    picked = (
        alt.Chart(rows.filter((pl.col("short") == pick["short"]) & (pl.col("long") == pick["long"])))
        .mark_rect(cornerRadius=6, fill=None, stroke=INK, strokeWidth=2)
        .encode(x="long:O", y=alt.Y("short:O", sort="descending"))
    )
    return (cells + labels + picked).properties(height=260)


def chart_header(title: str, key: str) -> bool:
    """A section title with the log-scale switch on the same line. Returns the switch state."""
    row = st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")
    row.subheader(title)
    return row.toggle("Log scale", value=True, key=key)


# The UI opens the crossover at 10/30, the setting the README and demo numbers use.
UI_DEFAULTS = {"moving_average_crossover": {"short_window": 10, "long_window": 30}}


def strategy_params(name: str) -> dict[str, int]:
    """A strategy's tunable int parameters and their defaults, read from its ``__init__`` signature."""
    params = inspect.signature(STRATEGIES[name].__init__).parameters.values()
    defaults = {p.name: p.default for p in params if type(p.default) is int}
    return defaults | UI_DEFAULTS.get(name, {})


def strategy_controls(bar, prefix: str) -> tuple[str, list[str]]:
    """Strategy picker plus one number input per parameter. Returns (name, ``--param`` flags)."""
    name = bar.selectbox(
        "Strategy", list(STRATEGIES), format_func=lambda n: STRATEGIES[n].label, width=190, key=f"{prefix}_strategy"
    )
    flags: list[str] = []
    for param, default in strategy_params(name).items():
        label = param.replace("_", " ").capitalize()
        value = bar.number_input(label, 1, 1000, default, width=110, key=f"{prefix}_{name}_{param}")
        flags += ["--param", f"{param}={value}"]
    return name, flags


def describe(argv: list[str]) -> str:
    a = parse_args(argv)
    what = html.escape(STRATEGIES[a.strategy].label)
    if a.param:
        what += f' <span class="num">{html.escape("/".join(str(v) for _, v in a.param))}</span>'
    # Rendered as HTML: escape anything that came from user input (defence in depth; symbols are validated too).
    return f"<strong>{what}</strong> on {html.escape(a.symbol)}, {a.start} to {a.end}, ${a.capital:,.0f} start"


def intro() -> None:
    """The "What am I looking at?" primer, open on a visitor's first page view."""
    with st.expander("What am I looking at?", expanded=not st.session_state.get("seen_intro")):
        st.session_state.seen_intro = True
        st.markdown(
            "A **backtest** replays a trading rule over past prices, day by day, as if you had followed it "
            "with real money. Here: Bitcoin, 2020 to 2024, $10,000 to start, and every trade pays fees and "
            "slippage.\n\n"
            "**Pick a strategy**, press **Run**, then read the numbers below. The grey dashed line is "
            "*buy & hold* (buy on day one, never sell): the bar any strategy has to clear.\n\n"
            + "\n".join(f"- **{cls.label}:** {cls.summary}" for cls in STRATEGIES.values())
            + "\n\n**Engine.** *Correct* only ever shows a strategy today and earlier. *Broken* secretly "
            "leaks tomorrow's prices, the bug this project is built to prevent. The **Leak test** page puts "
            "the two side by side.\n\n"
            "**Split.** Judges the strategy separately on the years after a date, so you can see if it "
            "holds up on data it wasn't tuned on."
        )


# --- Pages -----------------------------------------------------------------------


def page_backtest() -> None:
    st.title("Backtest")
    intro()
    bar = control_bar("bt")
    strategy, params = strategy_controls(bar, "bt")
    engine = bar.radio("Engine", ["Correct", "Broken"], horizontal=True, key="engine")
    split = bar.toggle("Split", key="bt_split", help="Also report in-sample / out-of-sample metrics around a date")
    split_date = bar.date_input("Split date", date(2023, 1, 1), disabled=not split, width=140, key="bt_split_date")
    base = data_and_costs(bar)
    clicked = bar.button("Run", type="primary", icon=":material/play_arrow:", key="bt_run")

    argv = [*base, "--strategy", strategy, *params, "--runner", engine.lower()]
    if split:
        argv += ["--split-date", str(split_date)]
    if clicked or "bt" not in st.session_state:  # result-first: the defaults run on first visit
        with st.spinner("Backtesting..."):
            st.session_state.bt = (run(argv), argv, engine == "Broken")

    out, shown_argv, broken = st.session_state.bt
    note(describe(shown_argv) + ("" if clicked or shown_argv == argv else ". Settings changed: press Run to update."))
    if broken:
        st.markdown(
            '<p class="note" style="color: var(--signal)">Broken engine: these numbers include look-ahead bias.</p>',
            unsafe_allow_html=True,
        )
    kpi_strip(out.metrics, "bt", out.benchmark, out.result.equity_curve["equity"])

    log_scale = chart_header("Equity", "bt_log")
    name = "Broken engine" if broken else "Strategy"
    st.altair_chart(
        equity_chart(
            {name: out.result.equity_curve, "Buy & hold": out.benchmark_result.equity_curve},
            {name: (SIGNAL if broken else INK, SOLID), "Buy & hold": (INK_3, DASHED)},
            log_scale,
        ),
        theme=None,
        width="stretch",
    )

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.subheader("Drawdown")
        st.altair_chart(drawdown_chart(out.result.equity_curve), theme=None, width="stretch")
        dd = out.metrics.max_drawdown
        if dd.peak:
            rec = f"recovered {dd.recovery:%Y-%m-%d}" if dd.recovery else "not recovered"
            note(
                f'Deepest: <span class="num">{pct(dd.depth)}</span>, {dd.peak:%Y-%m-%d} to {dd.trough:%Y-%m-%d}, {rec}'
            )
        if out.split:
            ins, oos = out.split
            st.subheader("In-sample vs out-of-sample")
            a, b = st.columns(2)
            with a, st.container(key="split-in"):
                st.metric(
                    "In-sample Sharpe", num(ins.sharpe_ratio), f"{pct(ins.total_return)} return", delta_color="off"
                )
            with b, st.container(key="split-out"):
                st.metric(
                    "Out-of-sample Sharpe", num(oos.sharpe_ratio), f"{pct(oos.total_return)} return", delta_color="off"
                )
    with right:
        trades = out.result.trades
        st.subheader(f"Trades ({trades.height})")
        st.dataframe(
            trades.select("timestamp", "side", "price", "quantity", "fee", "cash_after"),
            hide_index=True,
            height=300,
            width="stretch",
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Filled", format="YYYY-MM-DD"),
                "side": st.column_config.TextColumn("Side"),
                "price": st.column_config.NumberColumn("Price", format="dollar"),
                "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                "fee": st.column_config.NumberColumn("Fee", format="dollar"),
                "cash_after": st.column_config.NumberColumn("Cash after", format="dollar"),
            },
        )
    st.caption(f"MLflow run `{out.run_id}` · dataset `{out.dataset.dataset_id}`")


def page_leak() -> None:
    st.title("Leak test")
    note(
        "The same strategy on the same data, twice: once through the real engine, once through a copy "
        "that leaks one day of the future. Any gap between the two is pure cheating."
    )
    bar = control_bar("leak")
    strategy, params = strategy_controls(bar, "leak")
    base = data_and_costs(bar)
    clicked = bar.button("Run both", type="primary", icon=":material/compare_arrows:", key="leak_run")

    argv = [*base, "--strategy", strategy, *params]
    if clicked or "leak" not in st.session_state:
        with st.spinner("Running both engines..."):
            st.session_state.leak = (run([*argv, "--runner", "correct"]), run([*argv, "--runner", "broken"]))

    good, bad = st.session_state.leak
    g, b = good.metrics, bad.metrics
    st.html(
        f"""<div class="faceoff">
          <div><div class="who">Correct engine</div><div class="big">{pct(g.total_return)}</div>
            <div class="sub">Sharpe {num(g.sharpe_ratio)}, max drawdown {pct(g.max_drawdown.depth)}</div></div>
          <div class="broken"><div class="who">Broken engine (look-ahead leak)</div>
            <div class="big">{pct(b.total_return)}</div>
            <div class="sub">Sharpe {num(b.sharpe_ratio)}, max drawdown {pct(b.max_drawdown.depth)}</div></div>
        </div>"""
    )
    same = good.dataset.dataset_id == bad.dataset.dataset_id
    note(
        f'Same data (<span class="num">{good.dataset.dataset_id}</span>), same costs. Only the engine differs.'
        if same
        else "The two runs used different data."
    )

    log_scale = chart_header("Equity", "leak_log")
    st.altair_chart(
        equity_chart(
            {
                "Correct engine": good.result.equity_curve,
                "Broken engine": bad.result.equity_curve,
                "Buy & hold": good.benchmark_result.equity_curve,
            },
            {"Correct engine": (INK, SOLID), "Broken engine": (SIGNAL, SOLID), "Buy & hold": (INK_3, DASHED)},
            log_scale,
        ),
        theme=None,
        width="stretch",
    )

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.subheader("Side by side")
        engines = {"Correct engine": g, "Broken engine": b, "Buy & hold": good.benchmark}
        rows = [
            ("Total return", pct, "total_return"),
            ("CAGR", pct, "cagr"),
            ("Sharpe", num, "sharpe_ratio"),
            ("Win rate", lambda x: pct(x, signed=False), "win_rate"),
        ]
        table = [{"Metric": label, **{k: f(getattr(m, attr)) for k, m in engines.items()}} for label, f, attr in rows]
        table.append({"Metric": "Max drawdown", **{k: pct(m.max_drawdown.depth) for k, m in engines.items()}})
        st.dataframe(table, hide_index=True, width="stretch")
    with right:
        st.subheader("The leak")
        st.code(
            "# runner.py: rows 0..t, checked every bar\n"
            "history = history_up_to(data, t)\n\n"
            "# broken_runner.py: rows 0..t+1, tomorrow included\n"
            "history = data.slice(0, t + 2)",
            language="python",
        )
        note("One character apart. The future-poisoning test in tests/test_no_lookahead.py catches it.")


def logged_runs(tracking_uri: str) -> pl.DataFrame:
    """Every logged run, newest first. Not cached: a run made a moment ago must show up."""
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        return pl.DataFrame()
    runs = client.search_runs([exp.experiment_id], max_results=1000, order_by=["attributes.start_time DESC"])
    return pl.DataFrame(
        [
            {
                "started": datetime.fromtimestamp(r.info.start_time / 1000, UTC),
                "run": r.info.run_name,
                "runner": r.data.tags.get("runner", ""),
                "windows": "/".join(
                    filter(None, [r.data.params.get("short_window"), r.data.params.get("long_window")])
                ),
                "symbol": r.data.params.get("symbol", ""),
                "period": f"{r.data.params.get('start', '')} to {r.data.params.get('end', '')}",
                "capital": float(r.data.params.get("initial_capital", "nan")),
                "total_return": r.data.metrics.get("total_return"),
                "sharpe": r.data.metrics.get("sharpe_ratio"),
                "max_drawdown": r.data.metrics.get("max_drawdown"),
                "excess_return": r.data.metrics.get("excess_return"),
                "dataset": r.data.params.get("dataset_snapshot_id", ""),
                "sweep_id": r.data.params.get("sweep_id", ""),
                "run_id": r.info.run_id,
            }
            for r in runs
        ]
    )


@st.cache_data(show_spinner=False)
def run_equity(tracking_uri: str, run_id: str) -> pl.DataFrame | None:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = MlflowClient(tracking_uri=tracking_uri).download_artifacts(run_id, "equity_curve.csv", tmp)
        except Exception:
            return None
        return pl.read_csv(path, try_parse_dates=True)


def page_runs() -> None:
    st.title("Runs")
    runs = logged_runs(config.MLFLOW_TRACKING_URI)
    if runs.is_empty():
        note("No runs logged yet. Every run on the <strong>Backtest</strong> page is recorded here automatically.")
        return

    bar = control_bar("runs")
    runners = bar.multiselect("Engine", ["correct", "broken"], default=["correct", "broken"], width=260, key="runs_eng")
    datasets = sorted(d for d in runs["dataset"].unique().to_list() if d)
    dataset = bar.selectbox("Dataset fingerprint", ["All", *datasets], width=220, key="runs_ds")
    hide_sweeps = bar.toggle("Hide sweep runs", value=True, key="runs_hide")
    view = runs.filter(pl.col("runner").is_in(runners))
    if dataset != "All":
        view = view.filter(pl.col("dataset") == dataset)
    if hide_sweeps:
        view = view.filter(pl.col("sweep_id") == "")

    chart_slot = st.container()
    note(f'<span class="num">{view.height}</span> runs. Select up to {MAX_OVERLAY} rows to overlay their equity.')
    event = st.dataframe(
        view.drop("run_id", "sweep_id"),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "started": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm"),
            "total_return": st.column_config.NumberColumn("Total return", format="percent"),
            "max_drawdown": st.column_config.NumberColumn("Max DD", format="percent"),
            "excess_return": st.column_config.NumberColumn("vs B&H", format="percent"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "capital": st.column_config.NumberColumn("Capital", format="dollar"),
        },
    )
    rows = event.selection.rows[:MAX_OVERLAY]
    if not rows:
        return
    curves, styles = {}, {}
    for i, row in enumerate(view[rows].iter_rows(named=True)):
        label = f"{row['run']} {row['windows']} ({row['run_id'][:6]})"
        if (eq := run_equity(config.MLFLOW_TRACKING_URI, row["run_id"])) is not None:
            curves[label] = eq
            styles[label] = (SIGNAL if row["runner"] == "broken" else RUN_GREYS[i], RUN_DASHES[i])
    if curves:
        with chart_slot:
            log_scale = chart_header("Equity", "runs_log")
            st.altair_chart(equity_chart(curves, styles, log_scale, height=380), theme=None, width="stretch")


def page_sweep() -> None:
    st.title("Sweep")
    bar = control_bar("sweep")
    shorts = bar.text_input("Short windows", "5,10,20", width=150, key="sw_short")
    longs = bar.text_input("Long windows", "30,50,100", width=150, key="sw_long")
    split_date = bar.date_input("Split date", date(2023, 1, 1), width=140, key="sweep_split")
    base = data_and_costs(bar)
    clicked = bar.button("Run sweep", type="primary", icon=":material/grid_on:", key="sw_run")

    if clicked:
        try:
            s_list, l_list = [int(x) for x in shorts.split(",")], [int(x) for x in longs.split(",")]
        except ValueError:
            st.error("Windows must be comma-separated whole numbers, e.g. 5,10,20.")
            st.stop()
        n = sum(s < lg for s in s_list for lg in l_list)
        if n == 0:
            st.error("No valid pairs: every short window must be smaller than some long window.")
            st.stop()
        with st.spinner(f"Running {n} backtests..."):
            try:
                st.session_state.sweep = run_sweep(s_list, l_list, str(split_date), base)
            except (ValueError, TypeError) as e:
                st.error(str(e))
                st.stop()

    if (res := st.session_state.get("sweep")) is None:
        note(
            "Choosing settings by looking at results is its own kind of cheating. The sweep picks the winner on "
            "the <strong>in-sample</strong> period only, then shows how that pick did <strong>out of sample</strong>."
        )
        return

    sweep_id, rows = res
    pick = rows[0]
    note(
        f'Picked by in-sample Sharpe: <strong class="num">{pick["short"]}/{pick["long"]}</strong>. '
        f'Out of sample it ranked <strong class="num">{pick["oos_rank"]} of {len(rows)}</strong> '
        f'(Sharpe <span class="num">{pick["out_of_sample_sharpe"]:.2f}</span>). That is the honest number.'
    )
    df = pl.DataFrame(rows)
    values = df["in_sample_sharpe"].to_list() + df["out_of_sample_sharpe"].to_list()
    domain = [min(values), max(values)]  # one shared scale, so the two maps are comparable
    h1, h2 = st.columns(2, gap="large")
    h1.altair_chart(
        sharpe_heatmap(df, "in_sample_sharpe", "In-sample Sharpe", domain, pick), theme=None, width="stretch"
    )
    h2.altair_chart(
        sharpe_heatmap(df, "out_of_sample_sharpe", "Out-of-sample Sharpe", domain, pick), theme=None, width="stretch"
    )
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "in_sample_sharpe": st.column_config.NumberColumn("In-sample Sharpe", format="%.3f"),
            "out_of_sample_sharpe": st.column_config.NumberColumn("Out-of-sample Sharpe", format="%.3f"),
            "oos_rank": st.column_config.NumberColumn("OOS rank"),
        },
    )
    st.caption(f"MLflow filter: `params.sweep_id = '{sweep_id}'`")
