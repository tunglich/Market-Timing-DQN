"""Backtest 5 SB3 agents on TW50 trade_data.csv (2024-01-02 ~ 2026-03-31).

Compares against MVO baseline, ^TWII (Taiwan Weighted Index), 0050.TW (TW50 ETF).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

FINRL_ROOT = Path(__file__).resolve().parent.parent          # Market-Timing-DQN/finrl
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

OUT_DIR = FINRL_ROOT / "results/baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_DIR = str(FINRL_ROOT / "tw50_trained_models")
BUY_COST = 0.001425
SELL_COST = 0.004425
INITIAL_AMOUNT = 1_000_000
TURBULENCE_THRESHOLD = 150  # turbulence-based cap (no VIX in TW)

print("=== Loading data ===")
train = pd.read_csv("tw50_train_data.csv")
trade = pd.read_csv("tw50_trade_data.csv")
train = train.set_index(train.columns[0]); train.index.names = [""]
trade = trade.set_index(trade.columns[0]); trade.index.names = [""]
print(f"train rows={len(train):,}  dates={train['date'].min()} ~ {train['date'].max()}")
print(f"trade rows={len(trade):,}  dates={trade['date'].min()} ~ {trade['date'].max()}")
TRADE_START = trade["date"].min()
TRADE_END = trade["date"].max()

print("\n=== Loading trained agents ===")
def maybe_load(name, cls):
    path = os.path.join(TRAINED_MODEL_DIR, f"agent_{name}")
    if os.path.exists(path + ".zip"):
        try:
            m = cls.load(path)
            print(f"  loaded {name.upper()} from {path}.zip")
            return m
        except Exception as e:
            print(f"  FAILED to load {name.upper()}: {e}")
            return None
    print(f"  missing {name.upper()} at {path}.zip")
    return None

agents = {
    "a2c":  maybe_load("a2c",  A2C),
    "ppo":  maybe_load("ppo",  PPO),
    "ddpg": maybe_load("ddpg", DDPG),
    "td3":  maybe_load("td3",  TD3),
    "sac":  maybe_load("sac",  SAC),
}

stock_dim = len(trade.tic.unique())
state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
print(f"\nStock dim={stock_dim}  State space={state_space}")
env_kwargs = {
    "hmax": 100, "initial_amount": INITIAL_AMOUNT,
    "num_stock_shares": [0] * stock_dim,
    "buy_cost_pct": [BUY_COST] * stock_dim,
    "sell_cost_pct": [SELL_COST] * stock_dim,
    "state_space": state_space, "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS, "action_space": stock_dim,
    "reward_scaling": 1e-4,
}
# TW market: no VIX, use turbulence as risk indicator
e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=TURBULENCE_THRESHOLD,
                              risk_indicator_col="turbulence", **env_kwargs)

print("\n=== Running DRL agents ===")
results = {}
for name, model in agents.items():
    if model is None:
        continue
    try:
        df_av, df_act = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
        results[name] = df_av
        print(f"  {name.upper()}: final = ${float(df_av['account_value'].iloc[-1]):,.0f}")
        df_act.to_csv(OUT_DIR / f"{name}_actions.csv")
    except Exception as e:
        print(f"  {name.upper()} FAILED: {e}")

print("\n=== Computing MVO baseline ===")
def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")

StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)
arP = np.asarray(StockData)
R, C = arP.shape
arR = np.zeros([R - 1, C])
for j in range(C):
    for i in range(R - 1):
        arR[i, j] = (arP[i + 1, j] - arP[i, j]) / arP[i, j] * 100 if arP[i, j] > 0 else 0
meanR = np.mean(arR, axis=0); covR = np.cov(arR, rowvar=False)
from pypfopt.efficient_frontier import EfficientFrontier
ef = EfficientFrontier(meanR, covR, weight_bounds=(0, 0.5)); ef.max_sharpe()
w = ef.clean_weights()
mvo_w = np.array([INITIAL_AMOUNT * w[i] for i in range(len(w))])
last_price = np.array([1 / pp if pp > 0 else 0 for pp in StockData.tail(1).to_numpy()[0]])
init_p = mvo_w * last_price
MVO = pd.DataFrame(TradeData @ init_p, columns=["mvo"])
print(f"MVO final = ${float(MVO['mvo'].iloc[-1]):,.0f}")

print("\n=== TW benchmarks: ^TWII and 0050.TW ===")
benchmarks = {}
for sym, label in [("^TWII", "twii"), ("0050.TW", "0050")]:
    try:
        df_b = yf.download(sym, start=TRADE_START, end=TRADE_END, auto_adjust=False,
                           progress=False, threads=False, timeout=30)
        if df_b is None or df_b.empty:
            print(f"  {sym}: empty result, skipping")
            continue
        if df_b.columns.nlevels != 1:
            df_b.columns = df_b.columns.droplevel(1)
        df_b = df_b[["Close"]].reset_index()
        df_b.columns = ["date", "close"]
        df_b["date"] = df_b["date"].astype(str)
        fst = df_b["close"].iloc[0]
        s = df_b["close"].div(fst).mul(INITIAL_AMOUNT)
        benchmarks[label] = pd.Series(s.values, index=df_b["date"].values, name=label)
        print(f"  {sym}: {len(df_b)} rows, final = ${float(s.iloc[-1]):,.0f}")
    except Exception as e:
        print(f"  {sym} download FAILED: {e}")

def to_series(df):
    return df.set_index(df.columns[0])["account_value"]

combined = pd.DataFrame({n: to_series(d) for n, d in results.items()})
combined["mvo"] = MVO["mvo"]
for label, s in benchmarks.items():
    combined[label] = s

combined.to_csv(OUT_DIR / "backtest_results.csv")
print(f"\nResults saved to {OUT_DIR / 'backtest_results.csv'}")

def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 2:
        return {"final": float("nan"), "ret_pct": float("nan"), "sharpe": float("nan"), "mdd": float("nan")}
    ret = float(s.iloc[-1] / s.iloc[0] - 1) * 100
    daily = s.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else float("nan")
    cummax = s.cummax(); dd = (s - cummax) / cummax
    return {"final": float(s.iloc[-1]), "ret_pct": ret, "sharpe": sharpe, "mdd": float(dd.min()) * 100}

rows = [[c.upper(), *metrics(combined[c]).values()] for c in combined.columns]
metrics_df = pd.DataFrame(rows, columns=["agent", "final_$", "return_%", "sharpe", "max_dd_%"])
print("\n=== Backtest Metrics ===")
print(metrics_df.to_string(index=False, float_format="%.3f"))
metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)

plt.figure(figsize=(13, 7))
for col in combined.columns:
    if combined[col].notna().any():
        plt.plot(pd.to_datetime(combined.index), combined[col], label=col.upper(), linewidth=1.5)
plt.axhline(INITIAL_AMOUNT, color="grey", linewidth=0.8, linestyle="--", label=f"Initial ${INITIAL_AMOUNT:,}")
plt.title(f"FinRL TW50 Backtest  {TRADE_START} ~ {TRADE_END}  (Train end {train['date'].max()})")
plt.xlabel("Date"); plt.ylabel("Portfolio value (NTD)")
plt.legend(loc="best"); plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
png_path = OUT_DIR / "backtest_compare.png"
plt.savefig(png_path, dpi=130)
print(f"Plot saved to {png_path}")

summary = [
    "FinRL TW50 OOS Backtest",
    f"Train period: {train['date'].min()} ~ {train['date'].max()}",
    f"Trade period: {TRADE_START} ~ {TRADE_END}",
    f"Stocks: {stock_dim} ({sorted(trade.tic.unique().tolist())})",
    f"Initial capital: NT${INITIAL_AMOUNT:,}",
    f"Commissions: buy {BUY_COST*100:.4f}% / sell {SELL_COST*100:.4f}% (incl. 0.3% transaction tax)",
    f"Risk indicator: turbulence (threshold={TURBULENCE_THRESHOLD}, no VIX in TW)",
    "",
    "Final values & metrics:",
    metrics_df.to_string(index=False, float_format="%.3f"),
]
(OUT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n=== DONE ===")
