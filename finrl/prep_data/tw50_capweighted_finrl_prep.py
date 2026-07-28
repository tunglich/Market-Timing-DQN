"""Add FinRL default 8 technical indicators to existing TW50 50-tic data.

Uses existing tw50_market_weighted_*.pkl which already has:
  date, tic, open, high, low, close, des, cov_list, return_list

Output:
  tw50_capweighted_finrl_train.pkl  (train: 2006-01-09 ~ 2023-12-29)
  tw50_capweighted_finrl_trade.pkl  (trade: 2024-01-02 ~ 2026-03-31)

Indicators added (Variant A's feature set):
  macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

FINRL_ROOT = Path(__file__).resolve().parent.parent          # Market-Timing-DQN/finrl
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))

import numpy as np
import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer

INDICATORS_FINRL = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

print("=== Loading existing TW50 50-tic data ===")
train_old = pd.read_pickle("tw50_market_weighted_train_data.pkl")
trade_old = pd.read_pickle("tw50_market_weighted_trade_data.pkl")
print(f"train_old: {len(train_old):,} rows  {train_old.tic.nunique()} tics  "
      f"{train_old.date.min()} ~ {train_old.date.max()}")
print(f"trade_old: {len(trade_old):,} rows  {trade_old.tic.nunique()} tics  "
      f"{trade_old.date.min()} ~ {trade_old.date.max()}")

all_df = pd.concat([train_old, trade_old], ignore_index=True)
all_df = (all_df
          .drop_duplicates(subset=["date", "tic"], keep="first")
          .sort_values(["date", "tic"])
          .reset_index(drop=True))
print(f"Combined: {len(all_df):,} rows  {all_df.tic.nunique()} tics  "
      f"{all_df.date.min()} ~ {all_df.date.max()}")

print("\n=== Extracting cov_list/return_list per date (one per date) ===")
cov_by_date = all_df.groupby("date")["cov_list"].first().to_dict()
ret_by_date = all_df.groupby("date")["return_list"].first().to_dict()
print(f"  {len(cov_by_date):,} unique dates with cov_list")

ohlc_df = all_df[["date", "tic", "open", "high", "low", "close"]].copy()
if "volume" not in ohlc_df.columns:
    ohlc_df["volume"] = 1.0
ohlc_df = ohlc_df.sort_values(["date", "tic"]).reset_index(drop=True)

print("\n=== Running FinRL FeatureEngineer (technical indicators only) ===")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS_FINRL,
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False,
)
processed = fe.preprocess_data(ohlc_df)
processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)
print(f"Processed shape: {processed.shape}")
print(f"Tickers after FE: {processed.tic.nunique()}")
print(f"Cols: {list(processed.columns)}")

print("\n=== Sanity check: any NaN/inf in indicators? ===")
ind_cols = [c for c in INDICATORS_FINRL if c in processed.columns]
n_nan = processed[ind_cols].isna().sum().sum()
n_inf = np.isinf(processed[ind_cols].to_numpy(dtype=float)).sum()
print(f"  NaN total: {n_nan}  Inf total: {n_inf}")
if n_nan > 0:
    processed[ind_cols] = processed[ind_cols].fillna(0.0)
    print("  -> filled NaN with 0")
if n_inf > 0:
    processed[ind_cols] = processed[ind_cols].replace([np.inf, -np.inf], 0.0)
    print("  -> replaced Inf with 0")

print("\n=== Re-attaching cov_list & return_list & des ===")
processed["cov_list"] = processed["date"].map(cov_by_date)
processed["return_list"] = processed["date"].map(ret_by_date)
des_lookup = all_df.set_index(["date", "tic"])["des"].to_dict()
processed["des"] = list(map(des_lookup.get, zip(processed["date"], processed["tic"])))
print(f"  cov_list null: {processed['cov_list'].isna().sum()}")
print(f"  return_list null: {processed['return_list'].isna().sum()}")
print(f"  des null: {processed['des'].isna().sum()}")

if processed["cov_list"].isna().any():
    print("  Dropping rows without cov_list (warmup period before 252-day window)")
    processed = processed.dropna(subset=["cov_list"]).reset_index(drop=True)
    print(f"  After drop: {len(processed):,} rows")

print("\n=== Re-splitting into train/trade ===")
TRAIN_END = "2023-12-29"
TRADE_START = "2024-01-02"
train = processed[processed["date"] <= TRAIN_END].copy()
trade = processed[processed["date"] >= TRADE_START].copy()
print(f"train: {len(train):,} rows  {train.tic.nunique()} tics  "
      f"{train.date.min()} ~ {train.date.max()}")
print(f"trade: {len(trade):,} rows  {trade.tic.nunique()} tics  "
      f"{trade.date.min()} ~ {trade.date.max()}")

train_path = "tw50_capweighted_finrl_train.pkl"
trade_path = "tw50_capweighted_finrl_trade.pkl"
train.to_pickle(train_path)
trade.to_pickle(trade_path)
print(f"\nSaved: {train_path}, {trade_path}")
print("Indicators in saved data:", INDICATORS_FINRL)
print("\n=== Sample row ===")
print(train.head(2).to_string())
