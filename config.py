"""Project-wide configuration: paths, cost defaults and MLflow settings.

Plain typed constants, so every module imports one source of truth.
Nothing here needs secrets or environment variables: both data sources
(Binance Vision archives) and MLflow run locally with no API keys.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent

PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"  # <source>/symbol=<S>/year=<Y>/data.parquet
# data/sample/ has the same layout and is committed: Binance BTCUSDT 2020-2024, for offline runs.

# MLflow, fully local: run metadata (params, metrics, tags) in a SQLite file,
# artifacts (charts, CSVs) as plain files under ./mlruns. No server, no account.
# (MLflow 3.x refuses the older plain-directory store by default.)
MLFLOW_TRACKING_URI: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT: str = (PROJECT_ROOT / "mlruns").as_uri()

# --- Costs (there is no frictionless mode) ------------------------------------
# These defaults must stay strictly positive. The runner rejects rates <= 0.

DEFAULT_FEE_RATE: float = 0.001  # 10 bps of notional, charged on every fill
DEFAULT_SLIPPAGE_RATE: float = 0.0005  # 5 bps, always applied against the trader

DEFAULT_INITIAL_CAPITAL: float = 10_000.0

# Market impact (runner.CostModel): square-root law, sized from the previous
# IMPACT_LOOKBACK bars. The CLI turns these on; tiny accounts barely notice them,
# large ones do.
DEFAULT_IMPACT_COEFFICIENT: float = 1.0  # empirically of order 1
DEFAULT_MAX_PARTICIPATION: float = 0.1  # fill at most 10% of average daily dollar volume per bar
IMPACT_LOOKBACK: int = 20

# Fraction of available cash a BUY signal spends when it doesn't give a size.
# 1.0 = fully invested (the runner reserves room for the fee).
DEFAULT_BUY_FRACTION: float = 1.0

# --- Metrics -----------------------------------------------------------------

# Daily bars per year, by data source, used to annualize Sharpe ratio.
PERIODS_PER_YEAR: dict[str, int] = {
    "binance": 365,  # crypto trades 24/7
}
RISK_FREE_RATE: float = 0.0

# --- MLflow ------------------------------------------------------------------

# Runs from runner.py and broken_runner.py share this experiment, separated by
# the `runner` tag (`correct` | `broken`), so they can be compared side by side.
MLFLOW_EXPERIMENT_NAME: str = "backtests"

# --- Data sources ------------------------------------------------------------

# Binance Vision: bulk archives published under CC BY-NC-SA 4.0 (Binance Vision Dataset Terms v1.0).
BINANCE_VISION_URL: str = "https://data.binance.vision/data/spot"
