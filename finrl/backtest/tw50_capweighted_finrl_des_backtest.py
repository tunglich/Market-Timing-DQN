"""Backtest 5 cap-anchored agents (FinRL 8 + DES) on TW50.

Action: weights = softmax(log(cap_w) + action)
Indicators (9): macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30,
                close_30_sma, close_60_sma, des

Compares vs:
  - CAP_BH    : cap-weighted buy-and-hold using 2023-12-29 weights
  - 0050.TW   : market-cap weighted ETF
  - ^TWII     : Taiwan Weighted Index
  - MVO       : max-Sharpe portfolio re-anchored on train data
"""
from __future__ import annotations

import csv
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
DATA_DIR = FINRL_ROOT.parent / "data"                    # Market-Timing-DQN/data
TW50_LIST = DATA_DIR / "tw50_2023-12-29.csv"
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))
# Seed all RNGs for Figure 8 reproducibility (override via FINRL_SEED env var).
from finrl_seeds import read_seed_from_env, set_finrl_seeds
SEED = set_finrl_seeds(read_seed_from_env(default=42))


from finrl.agents.stablebaselines3.models import DRLAgent
from tw50_capweighted_env import CapAnchoredPortfolioEnv

OUT_DIR = FINRL_ROOT / "results/capweighted_finrl_des75"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_DIR = str(FINRL_ROOT / "tw50_capweighted_finrl_des_trained_models")
INITIAL_AMOUNT = 1_000_000
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30",
              "cci_30", "dx_30", "close_30_sma", "close_60_sma", "des"]
LAMBDA_DEV = 0.0
ACTION_CLIP = 3.0


def load_tw50_weights() -> dict[str, float]:
    weights: dict[str, float] = {}
    with TW50_LIST.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            tic = row[1].strip()
            w_str = row[4].strip().rstrip("%")
            try:
                weights[tic] = float(w_str)
            except ValueError:
                continue
    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}


print("=== Loading data ===")
train = pd.read_pickle("tw50_capweighted_finrl_des_train.pkl")
trade = pd.read_pickle("tw50_capweighted_finrl_des_trade.pkl")
train.index = train.date.factorize()[0]
trade.index = trade.date.factorize()[0]
print(f"train rows={len(train):,}  dates={train['date'].min()} ~ {train['date'].max()}")
print(f"trade rows={len(trade):,}  dates={trade['date'].min()} ~ {trade['date'].max()}")
TRADE_START = trade["date"].min()
TRADE_END = trade["date"].max()

cap_w = load_tw50_weights()
print(f"Loaded {len(cap_w)} cap-weights")
print(f"Top 5: {sorted(cap_w.items(), key=lambda x: -x[1])[:5]}")

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
    print(f"  missing {name.upper()} at {path}.zip")
    return None

agents = {
    "a2c":  maybe_load("a2c",  A2C),
    "ppo":  maybe_load("ppo",  PPO),
    "ddpg": maybe_load("ddpg", DDPG),
    "td3":  maybe_load("td3",  TD3),
    "sac":  maybe_load("sac",  SAC),
}

stock_dim = trade.tic.nunique()
state_space = stock_dim
env_kwargs = {
    "cap_weights": cap_w,
    "hmax": 100, "initial_amount": INITIAL_AMOUNT,
    "transaction_cost_pct": 0,
    "state_space": state_space, "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS, "action_space": stock_dim,
    "reward_scaling": 1e-4,
    "lambda_dev": LAMBDA_DEV,
    "action_clip": ACTION_CLIP,
}

print("\n=== Running DRL agents (cap-anchored, FinRL+DES indicators) ===")
results = {}
weights_memory = {}
for name, model in agents.items():
    if model is None:
        continue
    try:
        e_trade_gym = CapAnchoredPortfolioEnv(df=trade, **env_kwargs)
        df_dret, df_w = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
        df_dret = df_dret.copy()
        df_dret["date"] = pd.to_datetime(df_dret["date"])
        df_dret["account_value"] = (1 + df_dret["daily_return"]).cumprod() * INITIAL_AMOUNT
        results[name] = df_dret
        weights_memory[name] = df_w
        df_w.to_csv(OUT_DIR / f"{name}_weights.csv")
        print(f"  {name.upper()}: final = NT${float(df_dret['account_value'].iloc[-1]):,.0f}")
    except Exception as e:
        import traceback
        print(f"  {name.upper()} FAILED: {e}")
        traceback.print_exc()

print("\n=== Cap-weighted B&H baseline (2023-12-29 weights) ===")
trade_pivot = trade.pivot_table(index="date", columns="tic", values="close").sort_index()
first_prices = trade_pivot.iloc[0]
shares_per_ntd = pd.Series({t: cap_w.get(t, 0) / first_prices[t] if first_prices[t] > 0 else 0
                            for t in trade_pivot.columns})
cap_bh_value = trade_pivot.dot(shares_per_ntd) * INITIAL_AMOUNT
print(f"CAP_BH final = NT${float(cap_bh_value.iloc[-1]):,.0f}")

print("\n=== MVO baseline ===")
StockData = train.pivot_table(index="date", columns="tic", values="close")
TradeData = trade.pivot_table(index="date", columns="tic", values="close")
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
print(f"MVO final = NT${float(MVO['mvo'].iloc[-1]):,.0f}")

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
        print(f"  {sym}: {len(df_b)} rows, final = NT${float(s.iloc[-1]):,.0f}")
    except Exception as e:
        print(f"  {sym} download FAILED: {e}")

combined = pd.DataFrame()
for name, d in results.items():
    s = d.set_index(d["date"].dt.strftime("%Y-%m-%d"))["account_value"]
    combined[name] = s
combined["mvo"] = MVO["mvo"]
combined["cap_bh"] = pd.Series(cap_bh_value.values, index=cap_bh_value.index)
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
metrics_df = pd.DataFrame(rows, columns=["agent", "final_NT$", "return_%", "sharpe", "max_dd_%"])
print("\n=== Backtest Metrics ===")
print(metrics_df.to_string(index=False, float_format="%.3f"))
metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)

print("\n=== Weight diagnostics per agent ===")
diag_rows = []
for name, df_w in weights_memory.items():
    arr = df_w.to_numpy(dtype=float)
    eff_n = float(np.mean(1.0 / np.sum(arr ** 2, axis=1)))
    max_w = float(np.mean(np.max(arr, axis=1)))
    cap_arr = np.array([cap_w.get(t, 0.0) for t in df_w.columns])
    cap_arr = cap_arr / cap_arr.sum() if cap_arr.sum() > 0 else cap_arr
    avg_dev = float(np.mean(np.sum((arr - cap_arr) ** 2, axis=1)))
    turnover = float(np.mean(np.sum(np.abs(np.diff(arr, axis=0)), axis=1))) if len(arr) > 1 else 0.0
    diag_rows.append([name.upper(), eff_n, max_w, avg_dev, turnover])
diag_df = pd.DataFrame(diag_rows, columns=["agent", "eff_n", "max_w", "avg_dev_to_cap", "turnover"])
cap_arr_full = np.array([cap_w.get(t, 0.0) for t in next(iter(weights_memory.values())).columns]) \
    if weights_memory else np.array([])
if cap_arr_full.size > 0:
    cap_arr_full = cap_arr_full / cap_arr_full.sum()
    diag_df.loc[len(diag_df)] = ["CAP (ref)",
                                  float(1.0 / np.sum(cap_arr_full ** 2)),
                                  float(np.max(cap_arr_full)),
                                  0.0, 0.0]
print(diag_df.to_string(index=False, float_format="%.4f"))
diag_df.to_csv(OUT_DIR / "weight_diagnostics.csv", index=False)

plt.figure(figsize=(13, 7))
for col in combined.columns:
    if combined[col].notna().any():
        plt.plot(pd.to_datetime(combined.index), combined[col], label=col.upper(), linewidth=1.5)
plt.axhline(INITIAL_AMOUNT, color="grey", linewidth=0.8, linestyle="--", label=f"Initial NT${INITIAL_AMOUNT:,}")
plt.title(f"FinRL TW50 Cap-Anchored + FinRL 8 + DES (60K timesteps)  "
          f"{TRADE_START} ~ {TRADE_END}")
plt.xlabel("Date"); plt.ylabel("Portfolio value (NT$)")
plt.legend(loc="best", ncol=2); plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
png_path = OUT_DIR / "backtest_compare.png"
plt.savefig(png_path, dpi=130)
print(f"\nPlot saved to {png_path}")

summary = [
    "FinRL TW50 OOS Backtest -- Cap-Anchored + FinRL 8 indicators + DES",
    f"Train period: {train['date'].min()} ~ {train['date'].max()}",
    f"Trade period: {TRADE_START} ~ {TRADE_END}",
    f"Stocks: {stock_dim}",
    f"Initial capital: NT${INITIAL_AMOUNT:,}",
    f"Action: weights = softmax(log(cap_w) + action),  action in Box(+/-{ACTION_CLIP})",
    f"Training reward shaping: portfolio_value * (1 - lambda_dev * Sigma((w-cap)^2)),  lambda_dev=0.5",
    f"State: 252-day rolling cov_matrix (50x50) + indicators ({INDICATORS} 9x50)",
    f"Timesteps: 60,000 (3x of prior TW50 cap-anchored 20K default)",
    f"Transaction cost: NOT MODELED (FinRL portfolio env limitation)",
    f"CAP_BH: cap-weighted buy-and-hold using 2023-12-29 weights",
    "",
    "Final values & metrics:",
    metrics_df.to_string(index=False, float_format="%.3f"),
    "",
    "Weight diagnostics (avg over backtest period):",
    diag_df.to_string(index=False, float_format="%.4f"),
]
(OUT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n=== DONE ===")
