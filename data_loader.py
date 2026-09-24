"""Download, normalize, validate, store and load historical daily OHLCV data.

Responsibilities:
    * Fetch completed daily bars from Binance Vision (data.binance.vision), the
      bulk-download archive Binance publishes under CC BY-NC-SA 4.0 (Binance
      Vision Dataset Terms §3.1). No API key or account is needed.
    * Normalize them into one canonical Polars schema (``CANONICAL_SCHEMA``).
    * Validate that schema. An unsorted or duplicated time index can itself
      leak look-ahead, so validation raises on any violation.
    * Store bars as Parquet at ``<root>/<source>/symbol=<S>/year=<Y>/data.parquet``
      and load them back.
    * Compute the dataset identifier logged to MLflow with every run.

Only use data sources whose terms permit it; see README "Data and licensing".
"""

import hashlib
import io
import re
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import requests

from config import BINANCE_VISION_URL, PROCESSED_DATA_DIR

# Canonical bar schema. A daily bar's `timestamp` is its OPEN time: 00:00 UTC
# of the bar's date.
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "symbol": pl.String(),
}

_PARTITION_FILE = "data.parquet"
_VISION_PAUSE_S = 0.1  # between archive downloads (Binance Vision Dataset Terms §7)
_OHLC_REL_TOL = 1e-9  # float-rounding slack (see validate_canonical)


@dataclass(frozen=True)
class LoadedDataset:
    """A validated dataset ready for the runner, plus its provenance.

    Attributes:
        df: Canonical-schema daily bars for one symbol, sorted by ``timestamp``.
        dataset_id: Short content-based identifier (see ``compute_dataset_id``).
        source: ``"binance"``.
        symbol: Trading pair, e.g. ``"BTCUSDT"``.
        start: Date of the first bar actually loaded.
        end: Date of the last bar actually loaded.
    """

    df: pl.DataFrame
    dataset_id: str
    source: str
    symbol: str
    start: date
    end: date


# --- Fetching ----------------------------------------------------------------


def fetch_binance(symbol: str, start: date, end: date) -> pl.DataFrame:
    """Download completed daily bars for a Binance spot pair from Binance Vision.

    Finished months come from the monthly archives, the current month from the
    daily ones. Every archive is checked against the SHA-256 Binance publishes
    next to it. The range is inclusive on both ends.

    Licensing: the data is CC BY-NC-SA 4.0 (Binance Vision Dataset Terms §3.1):
    non-commercial use, and anything redistributed must credit Binance Vision
    and keep the same licence (§4.5).

    Raises:
        ValueError: if no archive exists for the range (e.g. unknown symbol) or a checksum fails.
    """
    validate_symbol(symbol)  # it becomes part of the download URL
    frames = [_read_klines(blob) for url in _archive_urls(symbol, start, end) if (blob := _download(url))]
    if not frames:
        raise ValueError(f"Binance Vision has no daily klines for {symbol!r} in {start}..{end}")

    # Archive rows: open_time, open, high, low, close, volume, close_time, ...
    # open_time is in milliseconds before 2025 and microseconds from 2025 on.
    open_time = pl.col("open_time")
    df = pl.concat(frames).select(
        pl.when(open_time > 10**14)
        .then(open_time)
        .otherwise(open_time * 1000)
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .alias("timestamp"),
        pl.col("open", "high", "low", "close", "volume"),
    )
    # Crypto trades every day, so a missing day means a missing archive, not a holiday.
    # A backtest would silently jump across the hole; refuse instead.
    if gaps := missing_days(df):
        raise ValueError(f"{len(gaps)} missing day(s) in the Binance Vision data for {symbol}, e.g. {gaps[:5]}")
    return _finalize(df, symbol, start, end)


def missing_days(df: pl.DataFrame) -> list[date]:
    """Calendar days between the first and last bar that have no bar."""
    days = df["timestamp"].dt.date()
    expected = pl.date_range(days.min(), days.max(), "1d", eager=True)
    return expected.filter(~expected.is_in(days.implode())).to_list()


FETCHERS: dict[str, Callable[[str, date, date], pl.DataFrame]] = {"binance": fetch_binance}


def _archive_urls(symbol: str, start: date, end: date) -> list[str]:
    """Monthly archive URLs for finished months, daily ones for the current month."""
    base = f"{BINANCE_VISION_URL}/{{}}/klines/{symbol}/1d/{symbol}-1d-{{}}.zip"
    today = datetime.now(UTC).date()
    this_month = today.replace(day=1)
    urls, month = [], start.replace(day=1)
    while month <= end:
        if month < this_month:
            urls.append(base.format("monthly", f"{month:%Y-%m}"))
        else:
            day = max(start, month)
            while day <= min(end, today - timedelta(days=1)):
                urls.append(base.format("daily", f"{day:%Y-%m-%d}"))
                day += timedelta(days=1)
        month = (month + timedelta(days=32)).replace(day=1)
    return urls


def _download(url: str) -> bytes | None:
    """Fetch one archive and verify its published SHA-256. ``None`` if it doesn't exist."""
    time.sleep(_VISION_PAUSE_S)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:  # before the pair was listed, or not published yet
        return None
    resp.raise_for_status()
    checksum = requests.get(url + ".CHECKSUM", timeout=30)
    checksum.raise_for_status()
    expected = checksum.text.split()[0]
    if hashlib.sha256(resp.content).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {url}")
    return resp.content


def _read_klines(zip_bytes: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        raw = zf.read(zf.namelist()[0])
    return pl.read_csv(
        raw,
        has_header=False,
        columns=[0, 1, 2, 3, 4, 5],
        new_columns=["open_time", "open", "high", "low", "close", "volume"],
        schema_overrides={"open_time": pl.Int64},
    ).with_columns(pl.col("open", "high", "low", "close", "volume").cast(pl.Float64))


def _finalize(df: pl.DataFrame, symbol: str, start: date, end: date) -> pl.DataFrame:
    """Cast to the canonical schema, keep only completed bars in range, and validate."""
    # Today's daily bar is still forming and its values will change. Storing it
    # would put a bar in the dataset that doesn't match the final history, so
    # only bars dated before today (UTC) are kept.
    today = datetime.now(UTC).date()
    df = (
        df.with_columns(pl.lit(symbol).alias("symbol"))
        .select([pl.col(name).cast(dtype) for name, dtype in CANONICAL_SCHEMA.items()])
        .filter(
            pl.col("timestamp").dt.date().is_between(start, end, closed="both"),
            pl.col("timestamp").dt.date() < today,
        )
    )
    if df.is_empty():
        raise ValueError(f"No completed bars for {symbol!r} in {start}..{end}")
    validate_canonical(df)
    return df


# --- Validation --------------------------------------------------------------


def validate_canonical(df: pl.DataFrame) -> None:
    """Raise ``ValueError`` unless ``df`` is a well-formed canonical bar frame.

    Checks, in order:
      * exact column names, order and dtypes from ``CANONICAL_SCHEMA``
        (so timestamps must be UTC-aware)
      * no nulls or NaNs
      * timestamps strictly increasing within each symbol, which rules out both
        duplicates and unsorted rows. A duplicate or out-of-order bar would let
        the runner's "rows 0..t" slice contain data from after bar t.
      * OHLC consistency: ``low <= open, close <= high`` (to within float
        rounding, relative 1e-9), and ``volume >= 0``
    """
    if df.schema != pl.Schema(CANONICAL_SCHEMA):
        raise ValueError(
            "DataFrame does not match the canonical schema.\n"
            f"  expected: {dict(CANONICAL_SCHEMA)}\n"
            f"  got:      {dict(df.schema)}"
        )

    null_counts = {c: n for c, n in df.null_count().row(0, named=True).items() if n}
    if null_counts:
        raise ValueError(f"Null values found (column -> count): {null_counts}")
    nan_rows = df.filter(pl.any_horizontal(pl.col(pl.Float64).is_nan()))
    if not nan_rows.is_empty():
        raise ValueError(f"NaN values found at timestamps: {_examples(nan_rows)}")

    prev_ts = pl.col("timestamp").shift(1).over("symbol")
    duplicates = df.filter(pl.col("timestamp") == prev_ts)
    if not duplicates.is_empty():
        raise ValueError(f"Duplicate timestamps found ({duplicates.height} rows), e.g. {_examples(duplicates)}")
    out_of_order = df.filter(pl.col("timestamp") < prev_ts)
    if not out_of_order.is_empty():
        raise ValueError(
            "Timestamps are not sorted ascending: "
            f"{out_of_order.height} rows are earlier than the row before them, e.g. {_examples(out_of_order)}"
        )

    # Allow float-rounding noise (relative 1e-9, far below a cent), catch any real inconsistency.
    lo_bound = pl.min_horizontal("open", "close") * (1 + _OHLC_REL_TOL)
    hi_bound = pl.max_horizontal("open", "close") * (1 - _OHLC_REL_TOL)
    bad_ohlc = df.filter((pl.col("low") > lo_bound) | (pl.col("high") < hi_bound) | (pl.col("volume") < 0))
    if not bad_ohlc.is_empty():
        raise ValueError(
            f"Inconsistent OHLCV bars ({bad_ohlc.height} rows; need low <= open,close <= high "
            f"and volume >= 0), e.g. {_examples(bad_ohlc)}"
        )


def _examples(rows: pl.DataFrame, n: int = 5) -> list[str]:
    return [str(ts) for ts in rows["timestamp"].head(n)]


# --- Storage -----------------------------------------------------------------


def write_partitioned(df: pl.DataFrame, source: str, root: Path = PROCESSED_DATA_DIR) -> list[Path]:
    """Write canonical bars to ``<root>/<source>/symbol=<S>/year=<Y>/data.parquet``.

    Writes are upserts. New bars are merged into any existing partition, and
    where timestamps collide the new bar wins. So fetching a narrower range
    later never deletes bars that were already stored. Each partition is
    written to a temp file, then atomically renamed into place.

    Returns:
        The partition files written, in (symbol, year) order.
    """
    _check_source(source)
    validate_canonical(df)

    parts = df.with_columns(pl.col("timestamp").dt.year().alias("year")).partition_by(
        ["symbol", "year"], as_dict=True, maintain_order=True
    )
    written: list[Path] = []
    for (symbol, year), part in sorted(parts.items()):
        path = _symbol_dir(root, source, str(symbol)) / f"year={year}" / _PARTITION_FILE
        path.parent.mkdir(parents=True, exist_ok=True)

        part = part.drop("year")
        if path.exists():
            # Existing rows first, so keep="last" lets the new rows win.
            part = (
                pl.concat([pl.read_parquet(path), part])
                .unique("timestamp", keep="last", maintain_order=True)
                .sort("timestamp")
            )
        validate_canonical(part)

        tmp = path.with_suffix(".parquet.tmp")
        part.write_parquet(tmp)
        tmp.replace(path)
        written.append(path)
    return written


def load(
    symbol: str,
    source: str,
    start: date | None = None,
    end: date | None = None,
    root: Path = PROCESSED_DATA_DIR,
) -> LoadedDataset:
    """Read one symbol's stored bars in ``[start, end]`` (inclusive; None = unbounded).

    Only the ``year=`` partitions that overlap the range are read, in year
    order. The result is validated, NOT re-sorted: if the store holds
    duplicate or out-of-order bars, this raises ``ValueError`` so the
    corruption gets noticed instead of silently patched over.

    Raises:
        FileNotFoundError: if no stored bars match.
        ValueError: if the stored data fails ``validate_canonical``.
    """
    lo, hi = (start.year if start else 0), (end.year if end else 9999)
    files = [f for f in partition_files(symbol, source, root) if lo <= int(f.parent.name.removeprefix("year=")) <= hi]
    if not files:
        raise FileNotFoundError(f"No stored data for {source}:{symbol} in range {start}..{end} under {root}")

    df = pl.concat([pl.read_parquet(f) for f in files])
    if start:
        df = df.filter(pl.col("timestamp").dt.date() >= start)
    if end:
        df = df.filter(pl.col("timestamp").dt.date() <= end)
    if df.is_empty():
        raise FileNotFoundError(f"No stored bars for {source}:{symbol} in range {start}..{end}")

    validate_canonical(df)

    return LoadedDataset(
        df=df,
        dataset_id=compute_dataset_id(df, source, symbol),
        source=source,
        symbol=symbol,
        start=df["timestamp"][0].date(),
        end=df["timestamp"][-1].date(),
    )


_SYMBOL = re.compile(r"[A-Z0-9]{2,20}")


def validate_symbol(symbol: str) -> None:
    """Raise ``ValueError`` unless ``symbol`` looks like a spot pair (``BTCUSDT``).

    Symbols become file paths and URLs, so anything else (``../``, slashes,
    markup) is rejected here, before it can escape the data directory.
    """
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"invalid symbol {symbol!r}: expected 2-20 capital letters or digits, e.g. BTCUSDT")


def partition_files(symbol: str, source: str, root: Path = PROCESSED_DATA_DIR) -> list[Path]:
    """One symbol's stored partition files, in year order."""
    _check_source(source)
    files = _symbol_dir(root, source, symbol).glob(f"year=*/{_PARTITION_FILE}")
    return sorted(files, key=lambda f: int(f.parent.name.removeprefix("year=")))


def _symbol_dir(root: Path, source: str, symbol: str) -> Path:
    validate_symbol(symbol)  # every read and write path goes through here
    return root / source / f"symbol={symbol}"


def _check_source(source: str) -> None:
    if source not in FETCHERS:
        raise ValueError(f"Unknown source {source!r}; expected one of {sorted(FETCHERS)}")


# --- Dataset identity --------------------------------------------------------


def compute_dataset_id(df: pl.DataFrame, source: str, symbol: str) -> str:
    """Short, deterministic identifier of exactly the bars a run used.

    SHA-256 over a header (source, symbol, first/last bar date, row count) plus
    the rows themselves serialized as CSV, truncated to 16 hex chars.

    The row contents are hashed along with the metadata on purpose. Data
    providers can revise history after the fact (Binance Vision's archives
    differ from its live API on a few days' volume, for example), so the same
    symbol and date range can hold different numbers at different times. A
    metadata-only id would label two different datasets as the same snapshot.
    The loaded rows are hashed rather than the Parquet file bytes, so the id
    doesn't depend on the writer version or compression settings, and a
    sub-year load doesn't hash rows outside its range.
    """
    first = df["timestamp"].min().date()
    last = df["timestamp"].max().date()
    header = f"{source}|{symbol}|{first}|{last}|{df.height}\n"
    digest = hashlib.sha256(header.encode())
    digest.update(df.write_csv().encode())
    return digest.hexdigest()[:16]
