"""Block committing data whose licence we don't have: anything from Yahoo Finance.

Yahoo's Terms of Service forbid redistributing its content (§2.8), and this repo
once leaked Yahoo data into its history. This check keeps it from happening again.
It fails (exit 1) if any given file:

* has "yahoo" or "yfinance" anywhere in its path, or
* is a data file whose columns look like Yahoo's (``Adj Close``, ``Stock Splits``,
  ``Dividends``, ``Capital Gains``), or whose bytes mention yahoo/yfinance
  (e.g. in Parquet metadata).

Usage:
    python scripts/check_data_provenance.py            # every tracked file (CI does this)
    python scripts/check_data_provenance.py FILE...    # specific files (the pre-commit hook)
"""

import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

BANNED_NAME = re.compile(r"yahoo|yfinance", re.I)
YAHOO_COLUMNS = {"adj close", "stock splits", "dividends", "capital gains"}
DATA_SUFFIXES = {".parquet", ".csv", ".tsv", ".json", ".feather", ".arrow", ".xlsx", ".pkl", ".zip"}


def problems(path: Path) -> list[str]:
    """Why ``path`` must not be committed; empty if it's fine."""
    found = []
    if BANNED_NAME.search(path.as_posix()):
        found.append("path mentions yahoo/yfinance")
    if path.suffix.lower() in DATA_SUFFIXES and path.is_file():
        if BANNED_NAME.search(path.read_bytes().decode("latin-1")):
            found.append("contents mention yahoo/yfinance")
        columns = _columns(path)
        if yahoo := sorted(YAHOO_COLUMNS & {c.strip().lower() for c in columns}):
            found.append(f"Yahoo-style columns {yahoo}")
    return found


def _columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pq.read_schema(path).names
    if suffix in {".csv", ".tsv"}:
        header = path.read_text(errors="replace").partition("\n")[0]
        return re.split(r"[,\t]", header)
    return []


def main(argv: list[str]) -> int:
    files = argv
    if not files:
        # -z / NUL-split: a path containing spaces must stay one path (and get checked).
        tracked = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True).stdout
        files = [f for f in tracked.split("\0") if f]
    bad = {f: p for f in files if (p := problems(Path(f)))}
    for f, why in bad.items():
        print(f"BLOCKED {f}: {'; '.join(why)}", file=sys.stderr)
    if bad:
        msg = "Yahoo Finance data may not be committed (Yahoo ToS §2.8). See README 'Data and licensing'."
        print(msg, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
