"""Prepare TW50 data for FinRL backtest 2024-01-02 ~ 2026-03-31.

Reads TW50 constituents (as of 2023-12-29) from d:\\DRL\\data\\tw50_2023-12-29.csv,
appends '.TW' suffix for yfinance, downloads with retries, applies FE
(no VIX since TWVIX not on yfinance, turbulence kept), splits train/trade.

Train:  2005-01-03 ~ 2024-01-01 (excl, i.e. through 2023-12-29)
Trade:  2024-01-02 ~ 2026-04-01 (excl, i.e. through 2026-03-31)

Outputs:
  d:\\DRL\\FinRL\\tw50_train_data.csv
  d:\\DRL\\FinRL\\tw50_trade_data.csv
"""
from __future__ import annotations

import csv
import itertools
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

FINRL_ROOT = Path(__file__).resolve().parent.parent          # Market-Timing-DQN/finrl
DATA_DIR = FINRL_ROOT.parent / "data"                    # Market-Timing-DQN/data
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))

from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

TW50_LIST = DATA_DIR / "tw50_2023-12-29.csv"

DOWNLOAD_START = "2005-01-03"
DOWNLOAD_END = "2026-04-01"
TRAIN_START = "2005-01-03"
TRAIN_END_EXCL = "2024-01-01"
TRADE_START = "2024-01-02"
TRADE_END_EXCL = "2026-04-01"

RETRIES = 5
SLEEP = 2.0


def load_tw50_tickers(csv_path: Path) -> list[str]:
    tics: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            tics.append(row[1].strip())
    return tics


def fetch_one(tic: str, start: str, end: str, retries: int = RETRIES) -> pd.DataFrame | None:
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(tic, start=start, end=end, auto_adjust=False,
                             progress=False, threads=False, timeout=30)
            if df is None or df.empty:
                print(f"  [{tic}] attempt {attempt}: empty result")
            else:
                if df.columns.nlevels != 1:
                    df.columns = df.columns.droplevel(1)
                df["tic"] = tic
                return df
        except Exception as e:
            print(f"  [{tic}] attempt {attempt} failed: {type(e).__name__}: {e}")
        if attempt < retries:
            time.sleep(SLEEP)
    return None


def fetch_all(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames, failed = [], []
    for i, tic in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] downloading {tic} ...")
        df = fetch_one(tic, start, end)
        if df is None:
            failed.append(tic)
        else:
            frames.append(df)
    if failed:
        print(f"\nWARNING: failed after retries: {failed}")
    if not frames:
        raise RuntimeError("All ticker downloads failed.")
    out = pd.concat(frames, axis=0).reset_index()
    out.rename(columns={
        "Date": "date", "Adj Close": "adjcp", "Close": "close",
        "High": "high", "Low": "low", "Volume": "volume", "Open": "open",
    }, inplace=True)
    out["adj"] = out["adjcp"] / out["close"]
    for col in ["open", "high", "low", "close"]:
        out[col] *= out["adj"]
    out = out.drop(["adjcp", "adj"], axis=1)
    out["day"] = out["date"].dt.dayofweek
    out["date"] = out["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    out = out.dropna().reset_index(drop=True)
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    print(f"\nDownloaded {len(out):,} rows for {out['tic'].nunique()} tickers "
          f"(retained: {sorted(out['tic'].unique())})")
    return out


print(f"=== Loading TW50 list from {TW50_LIST} ===")
tw50_codes = load_tw50_tickers(TW50_LIST)
tickers = [f"{c}.TW" for c in tw50_codes]
print(f"TW50 ({len(tickers)}): {tickers}")

print(f"\n=== Robust download: {DOWNLOAD_START} ~ {DOWNLOAD_END} (excl), retries={RETRIES} ===")
df_raw = fetch_all(tickers, DOWNLOAD_START, DOWNLOAD_END)
print(f"Date range: {df_raw['date'].min()} ~ {df_raw['date'].max()}")

print("\n=== Feature engineering (INDICATORS + turbulence, NO VIX for TW) ===")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=False,
    use_turbulence=True,
    user_defined_feature=False,
)
processed = fe.preprocess_data(df_raw)
print(f"After FE rows: {len(processed):,}")

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))
processed_full = (
    pd.DataFrame(combination, columns=["date", "tic"])
    .merge(processed, on=["date", "tic"], how="left")
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"]).fillna(0)
print(f"Full padded rows: {len(processed_full):,}")
print(f"Tickers retained: {processed_full['tic'].nunique()} "
      f"({sorted(processed_full['tic'].unique())})")

print("\n=== Splitting ===")
train = data_split(processed_full, TRAIN_START, TRAIN_END_EXCL)
trade = data_split(processed_full, TRADE_START, TRADE_END_EXCL)
print(f"Train rows: {len(train):,}  dates: {train['date'].min()} ~ {train['date'].max()}  "
      f"unique={train['date'].nunique()}")
print(f"Trade rows: {len(trade):,}  dates: {trade['date'].min()} ~ {trade['date'].max()}  "
      f"unique={trade['date'].nunique()}")

train.to_csv("tw50_train_data.csv")
trade.to_csv("tw50_trade_data.csv")
print("\nSaved: tw50_train_data.csv, tw50_trade_data.csv")
