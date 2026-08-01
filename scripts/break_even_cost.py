"""Compute break-even round-trip friction from backtest artefacts (Table 3).

Given a per-symbol backtest CSV (``backtest_summary*.csv`` with ``n_buy``,
``n_close`` columns) and a portfolio-level CSV (``portfolio_summary*.csv``
with ``dqn_final_pct``, ``bh_final_pct``, ``n_symbols``), report -- per
``(window, scheme)`` -- the extra bp/side that would erase the portfolio's
excess return over its buy-and-hold benchmark.

    trades_per_stock            = mean_i (n_buy_i + n_close_i)          # both sides
    portfolio_excess_bp         = 100 * (dqn_final_pct - bh_final_pct)
    break_even_bp_per_side      = portfolio_excess_bp / trades_per_stock

Usage:
    python scripts/break_even_cost.py \
        --backtest-summary backtest_summary_dow30.csv \
        --portfolio-summary portfolio_summary_dow30.csv \
        --out reports/break_even_dow30.csv

    python scripts/break_even_cost.py \
        --backtest-summary backtest_summary.csv \
        --portfolio-summary portfolio_summary.csv \
        --out reports/break_even_tw50.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def load_trades(backtest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(backtest_csv)
    need = {"symbol", "window", "n_buy", "n_close"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{backtest_csv}: missing columns {sorted(missing)}")
    df["total_sides"] = df["n_buy"].astype(int) + df["n_close"].astype(int)
    return df


def load_portfolio(portfolio_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(portfolio_csv)
    need = {"window", "scheme", "n_symbols", "dqn_final_pct", "bh_final_pct"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{portfolio_csv}: missing columns {sorted(missing)}")
    return df


def compute_break_even(trades: pd.DataFrame, port: pd.DataFrame) -> pd.DataFrame:
    per_window = (trades.groupby("window", as_index=False)
                  .agg(total_sides=("total_sides", "sum"),
                       n_stocks=("symbol", "nunique")))
    per_window["sides_per_stock"] = per_window["total_sides"] / per_window["n_stocks"]

    joined = port.merge(per_window[["window", "sides_per_stock", "n_stocks"]],
                        on="window", how="left")
    joined["excess_bp"] = 100.0 * (joined["dqn_final_pct"] - joined["bh_final_pct"])
    joined["break_even_bp_per_side"] = joined.apply(
        lambda r: (r["excess_bp"] / r["sides_per_stock"])
                  if pd.notna(r["sides_per_stock"]) and r["sides_per_stock"] > 0
                  else float("nan"),
        axis=1,
    )
    cols = ["universe", "window", "scheme", "n_symbols", "n_stocks",
            "dqn_final_pct", "bh_final_pct", "excess_bp",
            "sides_per_stock", "break_even_bp_per_side"]
    cols = [c for c in cols if c in joined.columns]
    return joined[cols].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backtest-summary", type=Path, required=True)
    ap.add_argument("--portfolio-summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    trades = load_trades(args.backtest_summary)
    port = load_portfolio(args.portfolio_summary)
    out = compute_break_even(trades, port)

    fmt = out.copy()
    for c in ("dqn_final_pct", "bh_final_pct", "excess_bp",
              "sides_per_stock", "break_even_bp_per_side"):
        if c in fmt.columns:
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    print(fmt.to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out.columns), lineterminator="\n")
            w.writeheader()
            for row in out.to_dict(orient="records"):
                w.writerow(row)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
