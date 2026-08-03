"""Portfolio-level backtest that aggregates the per-symbol DQN signals.

For each shipped `trained_models/<SYM>_all_<W>.data` checkpoint we replay the
same deterministic policy that `src/backtest.py` uses over the test window
2024-01-02 ~ 2026-03-30 and collect the environment's per-step reward (which
is percent-of-notional P&L for that symbol, including commission).

Two portfolio weighting variants are reported per window:

* **equal**  -- 1/N over every symbol that has a shipped checkpoint for that window.
* **cap**    -- TW-50 market-cap weights from `data/tw50_2023-12-29.csv`,
                renormalized over the covered symbols.

Buy&Hold reference is computed the same way, but using the raw close-to-close
daily return of each symbol.

Usage:

    python src/portfolio_backtest.py                                  # all windows, both schemes
    python src/portfolio_backtest.py --windows 65 75 --scheme equal
    python src/portfolio_backtest.py --universe tw50 --cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib import data, environ, models  # noqa: E402
from src.backtest import (  # noqa: E402
    TEST_START, TEST_END, WINDOWS,
    UNIVERSE_FILES, UNIVERSE_EXPECTED_COUNT,
    slice_test_csv, load_universe_symbols, resolve_model_path,
)

TRADING_DAYS_PER_YEAR = 252


def load_cap_weights(universe: str, data_dir: Path) -> dict[str, float]:
    """Return ``{symbol: raw_market_cap}`` (unnormalized). Cap-weight schemes
    renormalize over the actually-covered symbols downstream."""
    if universe != "tw50":
        raise SystemExit(f"cap weights not defined for universe={universe!r}")
    path = data_dir / UNIVERSE_FILES[universe]
    weights: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            sym = row[1].strip()
            weights[sym] = float(row[3])
    if len(weights) != UNIVERSE_EXPECTED_COUNT[universe]:
        raise SystemExit(f"{path}: parsed {len(weights)} rows, expected {UNIVERSE_EXPECTED_COUNT[universe]}")
    return weights


def load_price_weights(universe: str, data_dir: Path) -> dict[str, float]:
    """Return ``{symbol: reference_close_price}`` for the paper's price-weighted
    Dow-30 scheme (§3.4). Reference date is 2023-12-29 (test-window inception)."""
    if universe != "dow30":
        raise SystemExit(f"price weights only defined for universe=dow30, got {universe!r}")
    path = data_dir / "dow30_price_weights_2023-12-29.csv"
    if not path.is_file():
        raise SystemExit(f"missing price-weight file: {path}")
    weights: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            weights[row[1].strip()] = float(row[2])
    if len(weights) != UNIVERSE_EXPECTED_COUNT[universe]:
        raise SystemExit(f"{path}: parsed {len(weights)} rows, expected {UNIVERSE_EXPECTED_COUNT[universe]}")
    return weights


def replay_policy_rewards(model_path: Path, test_csv_tmp: Path,
                          commission_buy: float, commission_sell: float,
                          bars_count: int, device: str) -> np.ndarray:
    """Run the deterministic greedy policy and return the per-step reward array."""
    prices = {"P": data.load_relative(str(test_csv_tmp))}
    env = environ.StocksEnv(
        prices,
        bars_count=bars_count,
        commission=commission_buy,
        commission_buy=commission_buy,
        commission_sell=commission_sell,
        reset_on_close=False,
        reward_on_close=False,
        state_1d=True,
        volumes=False,
        random_ofs_on_reset=False,
    )
    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)
    net = models.DQNConv1DLarge(obs_shape, n_actions).to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()

    obs, _ = env.reset()
    rewards: list[float] = []
    with torch.no_grad():
        while True:
            obs_v = torch.tensor(np.array([obs], dtype=np.float32)).to(device)
            a = int(net(obs_v).max(dim=1)[1].item())
            obs, r, terminated, truncated, _ = env.step(a)
            rewards.append(float(r))
            if terminated or truncated:
                break
    return np.asarray(rewards, dtype=np.float64)


def build_symbol_series(df: pd.DataFrame, rewards: np.ndarray, bars_count: int
                        ) -> tuple[pd.Series, pd.Series]:
    """Return (dqn_daily_pct, bh_daily_pct) both indexed by trade DATE.

    Step ``i`` in the env corresponds to the transition ``bars_count + i`` to
    ``bars_count + i + 1``; we label the reward with the DATE of the second bar.
    """
    dates = pd.to_datetime(df["<DATE>"]).to_numpy()
    close = df["<CLOSE>"].to_numpy(dtype=np.float64)
    n_steps = len(rewards)
    step_dates = dates[bars_count + 1: bars_count + 1 + n_steps]
    dqn = pd.Series(rewards, index=pd.DatetimeIndex(step_dates), name="dqn")
    bh_ret = 100.0 * (close[bars_count + 1: bars_count + 1 + n_steps] /
                      close[bars_count: bars_count + n_steps] - 1.0)
    bh = pd.Series(bh_ret, index=pd.DatetimeIndex(step_dates), name="bh")
    return dqn, bh


def aggregate_portfolio(daily_by_sym: dict[str, pd.Series],
                        weights: dict[str, float] | None) -> pd.Series:
    """Aggregate {sym: daily_pct_series} into a single portfolio daily-pct series.

    ``weights=None`` means equal weight. Weights are renormalized per day over
    the symbols that have a value on that day."""
    if not daily_by_sym:
        return pd.Series(dtype=np.float64)
    frame = pd.concat(daily_by_sym, axis=1).sort_index()
    if weights is None:
        w_row = pd.DataFrame(1.0, index=frame.index, columns=frame.columns)
    else:
        w = pd.Series({s: float(weights.get(s, 0.0)) for s in frame.columns})
        w_row = pd.DataFrame(np.tile(w.values, (len(frame), 1)),
                             index=frame.index, columns=frame.columns)
    mask = frame.notna()
    w_row = w_row.where(mask, 0.0)
    row_sum = w_row.sum(axis=1).replace(0.0, np.nan)
    w_norm = w_row.div(row_sum, axis=0)
    return (frame.fillna(0.0) * w_norm).sum(axis=1)


def metrics_from_daily_pct(daily_pct: pd.Series) -> dict:
    """Compute {final_ret_%, ann_return_%, ann_vol_%, sharpe, max_dd_%} from a
    daily percent-return series. Compounded."""
    if daily_pct.empty:
        return {"final_ret_pct": np.nan, "ann_return_pct": np.nan,
                "ann_vol_pct": np.nan, "sharpe": np.nan, "max_dd_pct": np.nan}
    r = daily_pct.to_numpy(dtype=np.float64) / 100.0
    equity = np.cumprod(1.0 + r)
    final_ret_pct = 100.0 * (equity[-1] - 1.0)
    n = len(r)
    ann_return_pct = 100.0 * ((equity[-1]) ** (TRADING_DAYS_PER_YEAR / n) - 1.0) if n > 0 else np.nan
    ann_vol_pct = 100.0 * float(np.std(r, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR) if n > 1 else np.nan
    sharpe = (float(np.mean(r)) / float(np.std(r, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
              if n > 1 and float(np.std(r, ddof=1)) > 0 else np.nan)
    running_peak = np.maximum.accumulate(equity)
    max_dd_pct = 100.0 * float((equity / running_peak - 1.0).min())
    return {
        "final_ret_pct": float(final_ret_pct),
        "ann_return_pct": float(ann_return_pct) if np.isfinite(ann_return_pct) else np.nan,
        "ann_vol_pct": float(ann_vol_pct) if np.isfinite(ann_vol_pct) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_dd_pct": float(max_dd_pct),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=sorted(UNIVERSE_FILES), default="tw50")
    # `--accuracies` is the paper-facing name; `--windows` is the legacy alias.
    ap.add_argument("--accuracies", "--windows", type=int, nargs="+", choices=WINDOWS,
                    default=list(WINDOWS), dest="windows",
                    help="target directional accuracies rho in percent; legacy alias: --windows")
    ap.add_argument("--scheme", choices=("equal", "cap", "price", "both"), default="both",
                    help="Weighting scheme. 'both' expands to (equal, cap) for tw50 "
                         "and (equal, price) for dow30.")
    ap.add_argument("--models-dir", type=Path, default=REPO_ROOT / "trained_models")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--tmp-dir", type=Path, default=REPO_ROOT / "saves" / "_backtest_tmp")
    ap.add_argument("--bars", type=int, default=10)
    ap.add_argument("--commission-buy", type=float, default=0.1425,
                        help="Buy commission %% (default: 0.1425, TW retail broker fee, §3.3)")
    ap.add_argument("--commission-sell", type=float, default=0.4425,
                        help="Sell commission %% (default: 0.4425, broker fee + 0.30%% STT, §3.3)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-summary", type=Path,
                    default=REPO_ROOT / "portfolio_summary.csv")
    ap.add_argument("--out-timeseries", type=Path,
                    default=REPO_ROOT / "portfolio_timeseries.csv")
    args = ap.parse_args()

    if args.scheme == "cap" and args.universe != "tw50":
        raise SystemExit("--scheme cap is only supported for universe=tw50")
    if args.scheme == "price" and args.universe != "dow30":
        raise SystemExit("--scheme price is only supported for universe=dow30")

    device = "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    symbols = load_universe_symbols(args.universe, args.data_dir)
    cap_raw = load_cap_weights(args.universe, args.data_dir) if args.universe == "tw50" else None
    price_raw = load_price_weights(args.universe, args.data_dir) if args.universe == "dow30" else None

    schemes: list[str]
    if args.scheme == "both":
        schemes = ["equal", "cap"] if args.universe == "tw50" else ["equal", "price"]
    else:
        schemes = [args.scheme]

    def scheme_weights(scheme: str) -> dict[str, float] | None:
        if scheme == "equal":
            return None
        if scheme == "cap":
            return cap_raw
        if scheme == "price":
            return price_raw
        raise SystemExit(f"unknown scheme {scheme!r}")

    all_equity_curves: dict[str, pd.Series] = {}
    summary_rows: list[dict] = []

    for w in args.windows:
        dqn_by_sym: dict[str, pd.Series] = {}
        bh_by_sym: dict[str, pd.Series] = {}
        skipped: list[str] = []
        for sym in symbols:
            model_path = args.models_dir / f"{sym}_all_{w}.data"
            if not model_path.is_file():
                skipped.append(sym)
                continue
            csv_path = args.data_dir / f"{sym}_all_{w}.csv"
            if not csv_path.is_file():
                skipped.append(sym)
                continue
            tmp_csv = args.tmp_dir / f"{sym}_all_{w}_test.csv"
            df = slice_test_csv(csv_path, tmp_csv, TEST_START, TEST_END)
            rewards = replay_policy_rewards(
                model_path, tmp_csv,
                commission_buy=args.commission_buy,
                commission_sell=args.commission_sell,
                bars_count=args.bars, device=device,
            )
            dqn_series, bh_series = build_symbol_series(df, rewards, args.bars)
            dqn_by_sym[sym] = dqn_series
            bh_by_sym[sym] = bh_series
        n_used = len(dqn_by_sym)
        print(f"[window={w}] symbols_used={n_used}  skipped={len(skipped)}")

        for scheme in schemes:
            weights = scheme_weights(scheme)
            dqn_daily = aggregate_portfolio(dqn_by_sym, weights)
            bh_daily = aggregate_portfolio(bh_by_sym, weights)
            dqn_eq = (1.0 + dqn_daily / 100.0).cumprod()
            bh_eq = (1.0 + bh_daily / 100.0).cumprod()
            m_dqn = metrics_from_daily_pct(dqn_daily)
            m_bh = metrics_from_daily_pct(bh_daily)
            tag = f"{args.universe}_w{w}_{scheme}"
            all_equity_curves[f"dqn_{tag}"] = dqn_eq
            all_equity_curves[f"bh_{tag}"] = bh_eq
            summary_rows.append(dict(
                universe=args.universe, window=w, scheme=scheme,
                n_symbols=n_used,
                dqn_final_pct=round(m_dqn["final_ret_pct"], 3),
                dqn_ann_pct=round(m_dqn["ann_return_pct"], 3),
                dqn_sharpe=round(m_dqn["sharpe"], 3),
                dqn_max_dd_pct=round(m_dqn["max_dd_pct"], 3),
                bh_final_pct=round(m_bh["final_ret_pct"], 3),
                bh_ann_pct=round(m_bh["ann_return_pct"], 3),
                bh_sharpe=round(m_bh["sharpe"], 3),
                bh_max_dd_pct=round(m_bh["max_dd_pct"], 3),
                excess_pct=round(m_dqn["final_ret_pct"] - m_bh["final_ret_pct"], 3),
            ))
            print(f"  scheme={scheme:5s}  DQN final={m_dqn['final_ret_pct']:+7.2f}%  "
                  f"sharpe={m_dqn['sharpe']:+.2f}  maxDD={m_dqn['max_dd_pct']:+.2f}%   "
                  f"|  BH final={m_bh['final_ret_pct']:+7.2f}%  sharpe={m_bh['sharpe']:+.2f}  "
                  f"maxDD={m_bh['max_dd_pct']:+.2f}%   excess={m_dqn['final_ret_pct']-m_bh['final_ret_pct']:+.2f}%")

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.out_summary, index=False, lineterminator="\n")
    print(f"\nWrote summary: {args.out_summary}  ({len(summary_rows)} rows)")

    ts = pd.concat(all_equity_curves, axis=1).sort_index()
    ts.index.name = "date"
    ts.to_csv(args.out_timeseries, lineterminator="\n")
    print(f"Wrote timeseries: {args.out_timeseries}  ({len(ts)} rows x {ts.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
