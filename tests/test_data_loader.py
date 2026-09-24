"""Tests for ``data_loader``: fetch/store/load round trip, validation, dataset ids.

Tests marked ``network`` hit Binance Vision. Deselect them offline with
``pytest -m "not network"``. All the others use in-memory frames and ``tmp_path``.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import data_loader
from data_loader import CANONICAL_SCHEMA, compute_dataset_id, load, validate_canonical, write_partitioned
from tests.helpers import make_bars


def _bars(start: date, days: int) -> pl.DataFrame:
    """Daily bars from ``start``: close rises by 1 per day from 100."""
    return make_bars([100.0 + i for i in range(days)], start=start)


# --- Network round trip ------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize(("source", "symbol"), [("binance", "BTCUSDT")])
def test_fetch_save_reload_roundtrip(tmp_path, source: str, symbol: str) -> None:
    """Fetch a short range, save it, reload it: same data, same id on every load."""
    start, end = date(2024, 1, 1), date(2024, 3, 31)
    fetched = data_loader.FETCHERS[source](symbol, start, end)
    assert fetched.schema == pl.Schema(CANONICAL_SCHEMA)
    assert fetched.height > 0

    write_partitioned(fetched, source, root=tmp_path)
    first = load(symbol, source, start, end, root=tmp_path)
    second = load(symbol, source, start, end, root=tmp_path)

    assert_frame_equal(first.df, fetched)
    assert first.dataset_id == second.dataset_id
    assert len(first.dataset_id) == 16

    # A different range is a different dataset, so it gets a different id.
    narrower = load(symbol, source, start, date(2024, 2, 29), root=tmp_path)
    assert narrower.dataset_id != first.dataset_id


# --- Validation --------------------------------------------------------------


def test_validate_accepts_canonical_frame() -> None:
    validate_canonical(_bars(date(2024, 1, 1), 5))


def test_validate_rejects_unsorted_timestamps() -> None:
    df = _bars(date(2024, 1, 1), 5)
    shuffled = df[[0, 2, 1, 3, 4]]
    with pytest.raises(ValueError, match="not sorted"):
        validate_canonical(shuffled)


def test_validate_rejects_duplicate_timestamps() -> None:
    df = _bars(date(2024, 1, 1), 5)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_canonical(pl.concat([df[:3], df[2:]]))


def test_validate_rejects_naive_timestamps() -> None:
    df = _bars(date(2024, 1, 1), 5).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    with pytest.raises(ValueError, match="canonical schema"):
        validate_canonical(df)


def test_validate_tolerates_float_rounding_in_ohlc() -> None:
    """Real case seen with adjusted equity data: the adjusted high landed a few
    ULPs below the adjusted close. That's rounding, not bad data."""
    df = _bars(date(2024, 1, 1), 5).with_columns(
        pl.when(pl.int_range(pl.len()) == 2).then(pl.col("close") * (1 + 1e-14)).otherwise(pl.col("high")).alias("high")
    )
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.col("high") * (1 + 1e-14))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    validate_canonical(df)


def test_validate_rejects_real_ohlc_violation() -> None:
    """A close one cent above the high is a real inconsistency."""
    df = _bars(date(2024, 1, 1), 5).with_columns(
        pl.when(pl.int_range(pl.len()) == 2).then(pl.col("high") + 0.01).otherwise(pl.col("close")).alias("close")
    )
    with pytest.raises(ValueError, match="Inconsistent OHLCV"):
        validate_canonical(df)


def test_validate_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="canonical schema"):
        validate_canonical(_bars(date(2024, 1, 1), 5).drop("volume"))


# --- Storage -----------------------------------------------------------------


def test_load_empty_store_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load("TEST", "binance", root=tmp_path)


def test_write_is_upsert_not_overwrite(tmp_path) -> None:
    """Jan–Mar then Feb–Apr leaves Jan–Apr intact, with no duplicates."""
    write_partitioned(_bars(date(2024, 1, 1), 91), "binance", root=tmp_path)  # Jan 1 – Mar 31
    write_partitioned(_bars(date(2024, 2, 1), 90), "binance", root=tmp_path)  # Feb 1 – Apr 30

    ds = load("TEST", "binance", root=tmp_path)
    assert ds.start == date(2024, 1, 1)
    assert ds.end == date(2024, 4, 30)
    assert ds.df.height == 121  # every day, each exactly once


def test_partitions_by_year_and_prunes_on_load(tmp_path) -> None:
    written = write_partitioned(_bars(date(2023, 12, 1), 62), "binance", root=tmp_path)
    assert [p.parent.name for p in written] == ["year=2023", "year=2024"]
    assert all(p.parent.parent.name == "symbol=TEST" for p in written)

    ds = load("TEST", "binance", start=date(2024, 1, 1), root=tmp_path)
    assert (ds.start, ds.end) == (date(2024, 1, 1), date(2024, 1, 31))


def test_load_detects_corrupted_store(tmp_path) -> None:
    """If a partition file holds unsorted rows, load raises rather than silently sorting."""
    (written,) = write_partitioned(_bars(date(2024, 1, 1), 5), "binance", root=tmp_path)
    pl.read_parquet(written).reverse().write_parquet(written)
    with pytest.raises(ValueError, match="not sorted"):
        load("TEST", "binance", root=tmp_path)


# --- Dataset identity --------------------------------------------------------


def test_dataset_id_is_deterministic() -> None:
    df = _bars(date(2024, 1, 1), 30)
    assert compute_dataset_id(df, "binance", "TEST") == compute_dataset_id(df.clone(), "binance", "TEST")


def test_dataset_id_changes_with_data_or_metadata() -> None:
    df = _bars(date(2024, 1, 1), 30)
    base = compute_dataset_id(df, "binance", "TEST")

    restated = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 10).then(pl.col("close") + 0.01).otherwise(pl.col("close")).alias("close")
    )
    assert compute_dataset_id(restated, "binance", "TEST") != base
    assert compute_dataset_id(df, "other-source", "TEST") != base
    assert compute_dataset_id(df[:-1], "binance", "TEST") != base


def test_committed_sample_is_the_documented_snapshot() -> None:
    """demo_script.md and README.md quote numbers for this exact snapshot. If this
    fails, the sample changed and the docs' numbers and fingerprint need updating."""
    ds = load("BTCUSDT", "binance", root=Path(__file__).parent.parent / "data" / "sample")
    assert (ds.start, ds.end, ds.df.height) == (date(2020, 1, 1), date(2024, 12, 31), 1827)
    assert ds.dataset_id == "a58795bacb31c078"


@pytest.mark.parametrize("bad", ["../../../../etc", "BTC/USDT", "btcusdt", "<b>X</b>", "A", ""])
def test_symbols_that_could_escape_paths_or_urls_are_rejected(tmp_path, bad: str) -> None:
    """Review finding: a symbol like ../../../../etc resolved outside the data root."""
    with pytest.raises(ValueError, match="invalid symbol"):
        load(bad, "binance", root=tmp_path)
    with pytest.raises(ValueError, match="invalid symbol"):
        write_partitioned(
            _bars(date(2024, 1, 1), 3).with_columns(pl.lit(bad).alias("symbol")), "binance", root=tmp_path
        )


def test_missing_days_are_detected() -> None:
    """Review finding: a missing monthly archive was skipped silently, leaving a hole."""
    full = _bars(date(2024, 1, 1), 10)
    assert data_loader.missing_days(full) == []
    holed = full.filter(~pl.col("timestamp").dt.day().is_in([4, 5]))
    assert data_loader.missing_days(holed) == [date(2024, 1, 4), date(2024, 1, 5)]
