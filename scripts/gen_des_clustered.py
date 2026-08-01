"""Generate DES CSVs with clustered directional errors (Eq. 5 of the paper).

For each ticker we emit ``<ticker>_all_<accuracy>.csv`` under an output
directory that encodes ``beta`` and ``phi``, e.g.::

    data/clustered_b50_p30/AAPL_all_65.csv

The marginal accuracy of the resulting ``<DES>`` column matches the target
``accuracy`` (up to Monte-Carlo noise), but the flip mask is generated from a
single-factor latent model shared across all tickers so that misclassifications
concentrate in high-stress regimes:

    v_it = phi * v_{i, t-1} + sqrt(1 - phi**2) * eps_it,   eps_it ~ N(0, 1)
    z_it = beta * s_t + sqrt(1 - beta**2) * v_it
    flip = 1{z_it > Phi^{-1}(p)}
    DES  = y_it XOR flip     (0/1 encoding, agrees with y with prob p)

``s_t`` is the normal-score transform of a cross-sectional realized-volatility
stress index -- the 20-day rolling mean of the daily cross-sectional average of
absolute log returns. ``beta = phi = 0`` recovers the independent-flip
generator (``scripts/gen_des_by_accuracy.py``).

Usage:
    python scripts/gen_des_clustered.py --universe tw50 --accuracy 65 \
        --beta 0.5 --phi 0.3 --seed 42

    python scripts/gen_des_clustered.py --universe dow30 --accuracy 65 \
        --beta 0.5 --phi 0.3 --seed 42 --data-dir data
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm_ppf(p):
    """Vectorised inverse standard-normal CDF via ``scipy.stats.norm.ppf``."""
    return norm.ppf(np.asarray(p, dtype=float))
FUTURE_WINDOW = 20
OUTPUT_COLUMNS = ["<DATE>", "<DES>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOLUME>"]

UNIVERSE_FILES = {"tw50": "tw50_2023-12-29.csv", "dow30": "dow30_constituents.csv"}


def load_symbols(universe: str, data_dir: Path) -> list[str]:
    path = data_dir / UNIVERSE_FILES[universe]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[1].strip() for row in reader if row and row[0].strip().isdigit()]


def load_close_matrix(symbols: list[str], data_dir: Path) -> pd.DataFrame:
    """Load a wide-format Close DataFrame from the per-symbol OHLCV CSVs.

    We prefer the ``<sym>_all_75.csv`` files that already ship in ``data/``
    over the wide ``Open.csv``/``Close.csv`` matrices, so this script works for
    both the TW-50 and Dow-30 universes without extra prep."""
    frames: dict[str, pd.Series] = {}
    for sym in symbols:
        p = data_dir / f"{sym}_all_75.csv"
        if not p.is_file():
            raise SystemExit(f"missing per-symbol CSV for stress index: {p}")
        df = pd.read_csv(p, usecols=["<DATE>", "<CLOSE>"])
        df["<DATE>"] = pd.to_datetime(df["<DATE>"])
        frames[sym] = df.set_index("<DATE>")["<CLOSE>"].astype(float)
    close = pd.concat(frames, axis=1).sort_index()
    close = close.dropna(how="all")
    return close


def stress_index(close: pd.DataFrame, window: int = 20) -> pd.Series:
    """Normal-score-transformed realized-vol proxy shared by all symbols.

    Cross-sectional mean of |log returns|, smoothed by a ``window``-day rolling
    mean, then ranked and mapped through Phi^{-1} to yield a standard-normal
    series."""
    log_ret = np.log(close.where(close > 0)).diff()
    xs_abs = log_ret.abs().mean(axis=1)
    smoothed = xs_abs.rolling(window, min_periods=1).mean()
    ranks = smoothed.rank(method="average") / (smoothed.notna().sum() + 1.0)
    ranks = ranks.clip(1e-6, 1.0 - 1e-6)
    s = pd.Series(_norm_ppf(ranks.to_numpy()), index=smoothed.index, name="stress")
    s = s.fillna(0.0)
    return s


def compute_y(close: pd.Series, window: int = FUTURE_WINDOW) -> np.ndarray:
    n = len(close)
    v = close.to_numpy(dtype=float)
    future_mean = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(n, i + 1 + window)
        future_mean[i] = v[i + 1: end].mean()
    y = np.zeros(n, dtype=int)
    ok = ~np.isnan(future_mean)
    y[ok] = (future_mean[ok] >= v[ok]).astype(int)
    return y


def build_flip_matrix(index: pd.DatetimeIndex, s: pd.Series, symbols: list[str],
                      accuracy: int, beta: float, phi: float,
                      seed: int) -> pd.DataFrame:
    """Return a boolean ``flip`` DataFrame indexed by date and symbol.

    ``flip[t, i] = 1`` iff ``z_it > Phi^{-1}(p)``; then the DES for symbol ``i``
    on day ``t`` disagrees with the ground-truth label ``y_it``."""
    p = accuracy / 100.0
    kappa = float(_norm_ppf(np.asarray(p)))
    T, N = len(index), len(symbols)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(size=(T, N))
    v = np.zeros_like(eps)
    if not (0.0 <= phi < 1.0):
        raise SystemExit(f"--phi must be in [0, 1), got {phi}")
    if not (0.0 <= beta < 1.0):
        raise SystemExit(f"--beta must be in [0, 1), got {beta}")
    scale_eps = float(np.sqrt(1.0 - phi ** 2))
    v[0] = eps[0]
    for t in range(1, T):
        v[t] = phi * v[t - 1] + scale_eps * eps[t]
    s_arr = s.reindex(index).fillna(0.0).to_numpy()
    scale_v = float(np.sqrt(1.0 - beta ** 2))
    z = beta * s_arr[:, None] + scale_v * v
    flip = z > kappa
    return pd.DataFrame(flip, index=index, columns=symbols)


def build_and_write(sym: str, close_col: pd.Series, ohlcv_path: Path,
                    flip_col: pd.Series, out_dir: Path, accuracy: int) -> tuple[float, int]:
    df_full = pd.read_csv(ohlcv_path)
    df_full["<DATE>"] = pd.to_datetime(df_full["<DATE>"])
    y = compute_y(pd.Series(df_full["<CLOSE>"].astype(float).to_numpy(),
                            index=df_full["<DATE>"]))
    flip_aligned = flip_col.reindex(df_full["<DATE>"]).fillna(False).to_numpy().astype(bool)
    des = np.where(flip_aligned, 1 - y, y).astype(int)
    df_out = pd.DataFrame({
        "<DATE>": df_full["<DATE>"].dt.strftime("%Y-%m-%d"),
        "<DES>": des,
        "<OPEN>": df_full["<OPEN>"].to_numpy(),
        "<HIGH>": df_full["<HIGH>"].to_numpy(),
        "<LOW>": df_full["<LOW>"].to_numpy(),
        "<CLOSE>": df_full["<CLOSE>"].to_numpy(),
        "<VOLUME>": (df_full["<VOLUME>"].astype("int64").to_numpy()
                     if "<VOLUME>" in df_full.columns else np.zeros(len(df_full), dtype="int64")),
    }, columns=OUTPUT_COLUMNS)
    out_path = out_dir / f"{sym}_all_{accuracy}.csv"
    df_out.to_csv(out_path, index=False, lineterminator="\n")
    agree = float((des == y).mean())
    return agree, len(df_out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--universe", choices=sorted(UNIVERSE_FILES), required=True)
    ap.add_argument("--accuracy", type=int, required=True,
                    help="Target directional accuracy, integer in [50, 100]")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Cross-sectional common-factor loading (0=independent)")
    ap.add_argument("--phi", type=float, default=0.3,
                    help="Serial persistence of the idiosyncratic component")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Defaults to data/clustered_b<BB>_p<PP>/")
    ap.add_argument("--stress-window", type=int, default=20)
    args = ap.parse_args()
    if not (50 <= args.accuracy <= 100):
        raise SystemExit(f"--accuracy must be in [50, 100], got {args.accuracy}")

    if args.out_dir is None:
        tag = f"clustered_b{int(round(args.beta * 100)):02d}_p{int(round(args.phi * 100)):02d}"
        args.out_dir = args.data_dir / tag
    args.out_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_symbols(args.universe, args.data_dir)
    close = load_close_matrix(symbols, args.data_dir)
    s = stress_index(close, window=args.stress_window)
    flip = build_flip_matrix(close.index, s, symbols, args.accuracy,
                             args.beta, args.phi, args.seed)

    print(f"universe={args.universe}  T={len(close)}  N={len(symbols)}  "
          f"target_acc={args.accuracy}%  beta={args.beta}  phi={args.phi}")
    print(f"marginal agreement rate over all cells: "
          f"{100 * (1.0 - flip.to_numpy().mean()):.2f}%  (target {args.accuracy}%)")

    agreements: list[float] = []
    for sym in symbols:
        ohlcv_path = args.data_dir / f"{sym}_all_{args.accuracy}.csv"
        if not ohlcv_path.is_file():
            # Fall back to the 75%-accuracy file for the OHLCV columns.
            ohlcv_path = args.data_dir / f"{sym}_all_75.csv"
        agree, n = build_and_write(sym, close[sym], ohlcv_path, flip[sym],
                                   args.out_dir, args.accuracy)
        agreements.append(agree)
    print(f"Wrote {len(symbols)} files under {args.out_dir}")
    print(f"per-symbol DES-vs-y agreement: mean={100 * np.mean(agreements):.2f}%  "
          f"min={100 * np.min(agreements):.2f}%  max={100 * np.max(agreements):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
