"""Generate a `<ticker>_all_<accuracy>.csv` file for DRL training.

Given a stock ticker and a target label accuracy (integer in [50, 100]), this
script builds a CSV with columns `<DATE>,<DES>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOLUME>`
matching the format of existing files such as ``data/1101_all_60.csv``.

- OHLCV is loaded from wide-format matrices ``data/{Open,High,Low,Close,Volume}.csv``
  (rows = date, columns = ticker symbols).
- Ground-truth label ``y`` is 1 iff the mean of the *next* 20 trading days'
  ``<CLOSE>`` is >= today's ``<CLOSE>``; otherwise 0. For the very last row
  (no future data) ``y = 0``. Tail rows with fewer than 20 remaining days use
  the partial mean.
- ``<DES>`` is a noisy version of ``y`` with the given accuracy: each row keeps
  ``y`` with probability ``accuracy/100`` and flips to ``1 - y`` otherwise.

Usage:
    python scripts/gen_des_by_accuracy.py --ticker 1101 --accuracy 60 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_FILES = {
    "<OPEN>": "Open.csv",
    "<HIGH>": "High.csv",
    "<LOW>": "Low.csv",
    "<CLOSE>": "Close.csv",
    "<VOLUME>": "Volume.csv",
}
FUTURE_WINDOW = 20
OUTPUT_COLUMNS = ["<DATE>", "<DES>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOLUME>"]


def load_ohlcv(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load OHLCV for a single ticker from the wide-format matrices."""
    series = {}
    for col_name, fname in OHLCV_FILES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        df = pd.read_csv(path, index_col=0)
        if ticker not in df.columns:
            raise KeyError(
                f"Ticker '{ticker}' not found as a column in {path}. "
                f"First few columns: {list(df.columns[:5])}"
            )
        series[col_name] = df[ticker]

    out = pd.concat(series, axis=1)
    out.index = pd.to_datetime(out.index)
    out.index.name = "<DATE>"
    out = out.sort_index()
    # Drop rows where all OHLCV values are NaN (ticker not yet listed / delisted gap).
    out = out.dropna(how="all")
    # Drop any remaining rows with NaN in any OHLCV cell (mid-history gaps).
    out = out.dropna(how="any")
    return out


def compute_y(close: pd.Series, window: int = FUTURE_WINDOW) -> np.ndarray:
    """Return the ground-truth label vector.

    y[i] = 1 iff mean(close[i+1 : i+1+window]) >= close[i]. If there are no
    future rows available (last row), y = 0. If fewer than ``window`` future
    rows are available, the partial mean is used.
    """
    n = len(close)
    close_vals = close.to_numpy(dtype=float)
    # future_mean[i] = mean of close[i+1 .. i+window]
    future_mean = np.full(n, np.nan, dtype=float)
    for i in range(n - 1):
        end = min(n, i + 1 + window)
        future_mean[i] = close_vals[i + 1 : end].mean()
    y = np.zeros(n, dtype=int)
    valid = ~np.isnan(future_mean)
    y[valid] = (future_mean[valid] >= close_vals[valid]).astype(int)
    return y


def noisy_labels(y: np.ndarray, accuracy: int, rng: np.random.Generator) -> np.ndarray:
    """Return DES = y with probability accuracy/100, else 1 - y (per row)."""
    p = accuracy / 100.0
    keep = rng.random(len(y)) < p
    des = np.where(keep, y, 1 - y).astype(int)
    return des


def build_dataframe(ticker: str, accuracy: int, seed: int, data_dir: Path) -> pd.DataFrame:
    ohlcv = load_ohlcv(ticker, data_dir)
    y = compute_y(ohlcv["<CLOSE>"])
    rng = np.random.default_rng(seed)
    des = noisy_labels(y, accuracy, rng)

    out = pd.DataFrame(
        {
            "<DATE>": ohlcv.index.strftime("%Y-%m-%d"),
            "<DES>": des,
            "<OPEN>": ohlcv["<OPEN>"].to_numpy(),
            "<HIGH>": ohlcv["<HIGH>"].to_numpy(),
            "<LOW>": ohlcv["<LOW>"].to_numpy(),
            "<CLOSE>": ohlcv["<CLOSE>"].to_numpy(),
            "<VOLUME>": ohlcv["<VOLUME>"].astype("int64").to_numpy(),
        },
        columns=OUTPUT_COLUMNS,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. 1101")
    parser.add_argument(
        "--accuracy",
        type=int,
        required=True,
        help="Target label accuracy, integer in [50, 100]",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing Open.csv/High.csv/Low.csv/Close.csv/Volume.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --data-dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (50 <= args.accuracy <= 100):
        raise SystemExit(
            f"--accuracy must be an integer in [50, 100], got {args.accuracy}"
        )

    out_dir = args.out_dir if args.out_dir is not None else args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.ticker, args.accuracy, args.seed, args.data_dir)
    out_path = out_dir / f"{args.ticker}_all_{args.accuracy}.csv"
    df.to_csv(out_path, index=False)

    # Report an agreement rate as a sanity check.
    y = compute_y(
        pd.Series(df["<CLOSE>"].to_numpy(), index=pd.to_datetime(df["<DATE>"]))
    )
    match_rate = float((df["<DES>"].to_numpy() == y).mean())
    print(f"Wrote {out_path} ({len(df)} rows)")
    print(
        f"Target accuracy = {args.accuracy}%, actual DES vs y agreement = "
        f"{match_rate * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
