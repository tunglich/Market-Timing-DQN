"""TW50 cap-weighted prep: FinRL 8 indicators + DES (from all_75) + 252-day cov.

Source: data/<tic>_all_75.csv  (50 tickers from data/tw50_2023-12-29.csv)
        Columns: <DATE>,<DES>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOLUME>
        Pre-IPO zero OHLC backfilled with first non-zero close.

Output:
  tw50_capweighted_finrl_des_train.pkl  (2006-01-09 ~ 2023-12-29 ; first 252d skipped)
  tw50_capweighted_finrl_des_trade.pkl  (2024-01-02 ~ 2026-03-31)

Indicators (9):
  macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma, des
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

FINRL_ROOT = Path(__file__).resolve().parent.parent          # Market-Timing-DQN/finrl
DATA_DIR = FINRL_ROOT.parent / "data"                    # Market-Timing-DQN/data
TW50_LIST = DATA_DIR / "tw50_2023-12-29.csv"

os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))

import numpy as np
import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

TRAIN_START = "2005-01-03"
TRAIN_END_EXCL = "2024-01-01"
TRADE_START = "2024-01-02"
TRADE_END_EXCL = "2026-04-01"
LOOKBACK = 252

INDICATORS_FINRL = ["macd", "boll_ub", "boll_lb", "rsi_30",
                    "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
INDICATORS = INDICATORS_FINRL + ["des"]


def load_tw50_tickers(csv_path: Path) -> list[str]:
    tics: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            tics.append(row[1].strip())
    return tics


def load_one(ticker: str) -> pd.DataFrame:
    p = DATA_DIR / f"{ticker}_all_75.csv"
    df = pd.read_csv(p)
    df = df.rename(columns={
        "<DATE>": "date", "<DES>": "des", "<OPEN>": "open",
        "<HIGH>": "high", "<LOW>": "low", "<CLOSE>": "close", "<VOLUME>": "volume",
    })
    df["tic"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= TRAIN_START) & (df["date"] < TRADE_END_EXCL)]
    return df[["date", "tic", "open", "high", "low", "close", "volume", "des"]]


print(f"=== Loading TW50 list from {TW50_LIST} ===")
tickers = load_tw50_tickers(TW50_LIST)
print(f"TW50 ({len(tickers)}): {tickers}")

print(f"\n=== Reading {len(tickers)} *_all_75.csv files ===")
frames = [load_one(t) for t in tickers]
all_df = pd.concat(frames, axis=0).reset_index(drop=True)
all_df = all_df.sort_values(["date", "tic"]).reset_index(drop=True)
print(f"Combined rows: {len(all_df):,}, unique dates: {all_df['date'].nunique()}, "
      f"unique tickers: {all_df['tic'].nunique()}")

print("\n=== Backfilling pre-IPO OHLC zeros ===")
patched = []
for tic in sorted(all_df["tic"].unique()):
    sub = all_df[all_df["tic"] == tic].sort_values("date")
    nz = sub[sub["close"] > 0]
    if len(nz) == 0 or len(nz) == len(sub):
        continue
    first = nz.iloc[0]
    n_zero = (sub["close"] == 0).sum()
    mask = (all_df["tic"] == tic) & (all_df["close"] == 0)
    for col in ["open", "high", "low", "close"]:
        all_df.loc[mask, col] = float(first[col])
    patched.append((tic, int(n_zero), first["date"], float(first["close"])))
print(f"Patched {len(patched)} pre-IPO ticker(s).")
for tic, nz, d, p in patched[:10]:
    print(f"  {tic}: {nz:>4d} zero rows backfilled to {p:.2f} (first non-zero {d})")

print("\n=== Running FinRL FeatureEngineer (8 technical indicators) ===")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS_FINRL,
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False,
)
processed = fe.preprocess_data(all_df[["date", "tic", "open", "high", "low", "close", "volume"]])
processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)
print(f"Processed shape: {processed.shape}")
print(f"Tickers after FE: {processed.tic.nunique()}")

n_nan = processed[INDICATORS_FINRL].isna().sum().sum()
n_inf = np.isinf(processed[INDICATORS_FINRL].to_numpy(dtype=float)).sum()
print(f"  NaN total: {n_nan}  Inf total: {n_inf}")
if n_nan > 0:
    processed[INDICATORS_FINRL] = processed[INDICATORS_FINRL].fillna(0.0)
if n_inf > 0:
    processed[INDICATORS_FINRL] = processed[INDICATORS_FINRL].replace([np.inf, -np.inf], 0.0)

print("\n=== Re-attaching DES (per-(date, tic)) ===")
des_lookup = all_df.set_index(["date", "tic"])["des"].to_dict()
processed["des"] = list(map(des_lookup.get, zip(processed["date"], processed["tic"])))
processed["des"] = processed["des"].fillna(0).astype(float)
print(f"  des null after attach: {processed['des'].isna().sum()}")
print(f"  des distribution (first 10 tics): "
      f"{processed.groupby('tic')['des'].mean().head(10).to_dict()}")

print(f"\n=== Computing rolling {LOOKBACK}-day covariance per date ===")
processed.index = processed.date.factorize()[0]
cov_list = []
return_list = []
unique_dates = processed["date"].unique()
print(f"Unique trading dates: {len(unique_dates):,}")
t0 = time.time()
for i in range(LOOKBACK, len(unique_dates)):
    if i % 500 == 0 and i > LOOKBACK:
        elapsed = time.time() - t0
        eta = elapsed / (i - LOOKBACK) * (len(unique_dates) - i)
        print(f"  {i}/{len(unique_dates)} dates  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    sub = processed.loc[i - LOOKBACK:i, :]
    price_lb = sub.pivot_table(index="date", columns="tic", values="close")
    return_lb = price_lb.pct_change().dropna()
    covs = return_lb.cov().values
    cov_list.append(covs)
    return_list.append(return_lb)

df_cov = pd.DataFrame({"date": unique_dates[LOOKBACK:],
                       "cov_list": cov_list,
                       "return_list": return_list})
print(f"Cov rows: {len(df_cov):,}  ({unique_dates[LOOKBACK]} ~ {unique_dates[-1]})")

merged = processed.merge(df_cov, on="date")
merged = merged.sort_values(["date", "tic"]).reset_index(drop=True)
merged.index = merged.date.factorize()[0]
print(f"Merged rows: {len(merged):,}, unique dates: {merged['date'].nunique()}")

print("\n=== Splitting ===")
train = data_split(merged, TRAIN_START, TRAIN_END_EXCL)
trade = data_split(merged, TRADE_START, TRADE_END_EXCL)
train.index = train.date.factorize()[0]
trade.index = trade.date.factorize()[0]
print(f"Train rows: {len(train):,}  dates: {train['date'].min()} ~ {train['date'].max()}  "
      f"unique={train['date'].nunique()}")
print(f"Trade rows: {len(trade):,}  dates: {trade['date'].min()} ~ {trade['date'].max()}  "
      f"unique={trade['date'].nunique()}")

train.to_pickle("tw50_capweighted_finrl_des_train.pkl")
trade.to_pickle("tw50_capweighted_finrl_des_trade.pkl")
print("\nSaved: tw50_capweighted_finrl_des_train.pkl, tw50_capweighted_finrl_des_trade.pkl")
print(f"Indicators ({len(INDICATORS)}): {INDICATORS}")
