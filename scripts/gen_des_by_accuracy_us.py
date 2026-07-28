"""Generate ``<ticker>_all_<accuracy>.csv`` for a US stock (Dow 30 style).

Mirrors ``scripts/gen_des_by_accuracy.py`` but:

- Downloads OHLCV from Yahoo Finance with ``auto_adjust=True`` (matches the
  existing ``data/AAPL_all_*.csv`` style, where prices are split+dividend
  adjusted).
- Emits US column order: ``<DATE>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOLUME>,<DES>``
  (DES **last**, unlike the TW50 files which have DES second).

Ground-truth ``y`` and noisy ``<DES>`` follow the same rules as the TW50
generator: ``y[i] = 1`` iff mean(next 20 closes) >= today's close (partial
mean for the tail; ``y = 0`` for the very last row).

Usage:
    python scripts/gen_des_by_accuracy_us.py --ticker V --accuracy 75 --seed 42
    # or generate all four accuracies at once:
    python scripts/gen_des_by_accuracy_us.py --ticker V --accuracies 55 60 65 75
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FUTURE_WINDOW = 20
OUTPUT_COLUMNS = ["<DATE>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOLUME>", "<DES>"]

DEFAULT_START = "2005-01-03"
DEFAULT_END = "2026-03-31"          # inclusive of 2026-03-30 (yfinance end is exclusive)


def download_ohlcv(ticker: str, start: str, end: str, retries: int = 5, sleep: float = 2.0) -> pd.DataFrame:
    """Download adjusted OHLCV via yfinance with retries.

    Returns a DataFrame indexed by date with columns
    ``[<OPEN>, <HIGH>, <LOW>, <CLOSE>, <VOLUME>]``.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False, threads=False, timeout=30,
            )
            if df is None or df.empty:
                raise RuntimeError(f"empty result for {ticker}")
            if df.columns.nlevels != 1:
                df.columns = df.columns.droplevel(1)
            df = df.rename(columns={
                "Open": "<OPEN>", "High": "<HIGH>", "Low": "<LOW>",
                "Close": "<CLOSE>", "Volume": "<VOLUME>",
            })
            df = df[["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOLUME>"]]
            df = df.dropna(how="any")
            df.index = pd.to_datetime(df.index)
            df.index.name = "<DATE>"
            return df
        except Exception as e:                                                # noqa: BLE001
            last_err = e
            print(f"  [{ticker}] attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(sleep)
    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts: {last_err}")


def compute_y(close: pd.Series, window: int = FUTURE_WINDOW) -> np.ndarray:
    """y[i] = 1 iff mean(close[i+1 .. i+window]) >= close[i]. Last row -> 0."""
    n = len(close)
    close_vals = close.to_numpy(dtype=float)
    future_mean = np.full(n, np.nan, dtype=float)
    for i in range(n - 1):
        end = min(n, i + 1 + window)
        future_mean[i] = close_vals[i + 1 : end].mean()
    y = np.zeros(n, dtype=int)
    valid = ~np.isnan(future_mean)
    y[valid] = (future_mean[valid] >= close_vals[valid]).astype(int)
    return y


def noisy_labels(y: np.ndarray, accuracy: int, rng: np.random.Generator) -> np.ndarray:
    """Keep y with probability accuracy/100, else flip. Applied per row."""
    p = accuracy / 100.0
    keep = rng.random(len(y)) < p
    return np.where(keep, y, 1 - y).astype(int)


def build_dataframe(ohlcv: pd.DataFrame, accuracy: int, seed: int) -> pd.DataFrame:
    y = compute_y(ohlcv["<CLOSE>"])
    rng = np.random.default_rng(seed)
    des = noisy_labels(y, accuracy, rng)
    return pd.DataFrame(
        {
            "<DATE>": ohlcv.index.strftime("%Y-%m-%d"),
            "<OPEN>": ohlcv["<OPEN>"].to_numpy(),
            "<HIGH>": ohlcv["<HIGH>"].to_numpy(),
            "<LOW>": ohlcv["<LOW>"].to_numpy(),
            "<CLOSE>": ohlcv["<CLOSE>"].to_numpy(),
            "<VOLUME>": ohlcv["<VOLUME>"].astype("int64").to_numpy(),
            "<DES>": des,
        },
        columns=OUTPUT_COLUMNS,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ticker", required=True, help="US ticker symbol, e.g. V")
    p.add_argument("--accuracy", type=int, default=None,
                   help="Single accuracy target in [50,100].")
    p.add_argument("--accuracies", type=int, nargs="+", default=None,
                   help="Multiple accuracy targets, e.g. --accuracies 55 60 65 75.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    p.add_argument("--start", default=DEFAULT_START, help=f"Download start (default {DEFAULT_START}).")
    p.add_argument("--end", default=DEFAULT_END,
                   help=f"Download end EXCLUSIVE (default {DEFAULT_END}).")
    p.add_argument("--out-dir", type=Path, default=Path("data"),
                   help="Output directory (default: data/).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.accuracy is None and not args.accuracies:
        raise SystemExit("Provide --accuracy or --accuracies.")
    accs = args.accuracies if args.accuracies else [args.accuracy]
    for a in accs:
        if not (50 <= a <= 100):
            raise SystemExit(f"accuracy must be in [50, 100], got {a}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.ticker} from {args.start} to {args.end} (exclusive) ...")
    ohlcv = download_ohlcv(args.ticker, args.start, args.end)
    print(f"  got {len(ohlcv)} rows, {ohlcv.index[0].date()} .. {ohlcv.index[-1].date()}")

    for a in accs:
        df = build_dataframe(ohlcv, a, args.seed)
        out_path = args.out_dir / f"{args.ticker}_all_{a}.csv"
        df.to_csv(out_path, index=False)
        # Report actual DES-vs-y agreement rate.
        y = compute_y(pd.Series(df["<CLOSE>"].to_numpy(),
                                index=pd.to_datetime(df["<DATE>"])))
        match_rate = float((df["<DES>"].to_numpy() == y).mean())
        print(f"Wrote {out_path} ({len(df)} rows)  "
              f"target={a}%  actual DES-vs-y agreement={match_rate * 100:.2f}%")


if __name__ == "__main__":
    main()
