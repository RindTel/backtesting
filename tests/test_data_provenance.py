"""The guardrail that keeps Yahoo Finance data out of the repo (scripts/check_data_provenance.py)."""

import subprocess

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.check_data_provenance import main, problems
from tests.helpers import make_bars


def _dir(tmp_path_factory):
    """A neutrally named temp dir: pytest names tmp_path after the test, and these test names say 'yahoo'."""
    return tmp_path_factory.mktemp("case")


def test_blocks_yahoo_paths_even_in_our_own_format(tmp_path_factory) -> None:
    """How the old leak looked: canonical columns, but stored under a yfinance/ directory."""
    f = _dir(tmp_path_factory) / "data" / "sample" / "yfinance" / "symbol=SPY" / "year=2020" / "data.parquet"
    f.parent.mkdir(parents=True)
    make_bars([100.0, 101.0], symbol="SPY").write_parquet(f)
    assert problems(f) == ["path mentions yahoo/yfinance"]


def test_blocks_parquet_with_yahoo_columns(tmp_path_factory) -> None:
    f = _dir(tmp_path_factory) / "prices.parquet"
    pl.DataFrame({"Close": [1.0], "Adj Close": [1.0], "Stock Splits": [0.0]}).write_parquet(f)
    assert problems(f) == ["Yahoo-style columns ['adj close', 'stock splits']"]


def test_blocks_csv_with_yahoo_header(tmp_path_factory) -> None:
    f = _dir(tmp_path_factory) / "prices.csv"
    f.write_text("Date,Open,High,Low,Close,Volume,Dividends\n2024-01-02,1,1,1,1,1,0\n")
    assert problems(f) == ["Yahoo-style columns ['dividends']"]


def test_blocks_yahoo_mentioned_in_file_metadata(tmp_path_factory) -> None:
    f = _dir(tmp_path_factory) / "prices.parquet"
    table = pa.table({"close": [1.0]}).replace_schema_metadata({"source": "query1.finance.yahoo.com"})
    pq.write_table(table, f)
    assert problems(f) == ["contents mention yahoo/yfinance"]


def test_allows_clean_data(tmp_path_factory) -> None:
    f = _dir(tmp_path_factory) / "data.parquet"
    make_bars([100.0, 101.0], symbol="BTCUSDT").write_parquet(f)
    assert problems(f) == []


def test_every_tracked_file_passes() -> None:
    """What CI runs: the committed repo (including data/sample) is clean."""
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    assert tracked, "no tracked files: not in a git checkout?"
    assert main([]) == 0


def test_paths_with_spaces_are_checked_as_one_path(tmp_path_factory) -> None:
    """Review finding: splitting `git ls-files` on whitespace turned one path with a space
    into two nonexistent ones, so the real file was never checked."""
    f = _dir(tmp_path_factory) / "my prices.csv"
    f.write_text("Date,Adj Close\n2024-01-02,1\n")
    assert main([str(f)]) == 1
