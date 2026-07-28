"""Walk-forward variant of tw50_capweighted_finrl_des_train.py.

Uses expanding-window walk-forward validation (4 folds over 5 chronological
segments of the 2005-01-03 .. 2023-12-29 sample period). The final fold's
model is saved to the standard trained_models directory so the existing
backtest script (tw50_capweighted_finrl_des_backtest.py) can load it
unmodified.

  Fold 1: train = S1,         val = S2
  Fold 2: train = S1..S2,     val = S3
  Fold 3: train = S1..S3,     val = S4
  Fold 4: train = S1..S4,     val = S5   <-- model used for 2024-2026 test

Timesteps per fold default to 15_000 (~= 60_000 total across 4 folds,
matching the single-shot training budget). Override with --timesteps.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

FINRL_ROOT = Path(__file__).resolve().parent.parent          # Market-Timing-DQN/finrl
DATA_DIR = FINRL_ROOT.parent / "data"                    # Market-Timing-DQN/data
TW50_LIST = DATA_DIR / "tw50_2023-12-29.csv"
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))

import pandas as pd

from tw50_capweighted_env import CapAnchoredPortfolioEnv
from walk_forward import run_walk_forward

LAMBDA_DEV = 0.5
ACTION_CLIP = 3.0

TRAINED_MODEL_DIR = FINRL_ROOT / "tw50_capweighted_finrl_des_trained_models"
RESULTS_DIR = FINRL_ROOT / "tw50_capweighted_finrl_des_walkforward_results"
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30",
              "cci_30", "dx_30", "close_30_sma", "close_60_sma", "des"]

PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 128}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {"batch_size": 128, "buffer_size": 100000, "learning_rate": 0.0001,
              "learning_starts": 100, "ent_coef": "auto_0.1"}

AGENT_CONFIGS = {
    "a2c": None,
    "ppo": PPO_PARAMS,
    "ddpg": None,
    "td3": TD3_PARAMS,
    "sac": SAC_PARAMS,
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=15_000,
                        help="Training timesteps per fold per agent (default 15000).")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of chronological segments (default 5 -> 4 folds).")
    parser.add_argument("--train-pkl", type=str,
                        default="tw50_capweighted_finrl_des_train.pkl",
                        help="Pickled training dataframe (relative to finrl/ dir).")
    args = parser.parse_args()

    print(f"=== Loading {args.train_pkl} ===")
    train = pd.read_pickle(args.train_pkl)
    train.index = train.date.factorize()[0]
    print(f"train rows={len(train):,}  tickers={train.tic.nunique()}  "
          f"dates={train['date'].min()} ~ {train['date'].max()}")

    cap_w = load_tw50_weights()
    print(f"Loaded {len(cap_w)} cap-weights from {TW50_LIST.name}")

    stock_dim = train.tic.nunique()
    state_space = stock_dim
    env_kwargs = {
        "cap_weights": cap_w,
        "hmax": 100,
        "initial_amount": 1_000_000,
        "transaction_cost_pct": 0,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
        "lambda_dev": LAMBDA_DEV,
        "action_clip": ACTION_CLIP,
    }

    def env_ctor(df: pd.DataFrame) -> CapAnchoredPortfolioEnv:
        return CapAnchoredPortfolioEnv(df=df, **env_kwargs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = run_walk_forward(
        train_df=train,
        env_ctor=env_ctor,
        agent_configs=AGENT_CONFIGS,
        timesteps_per_fold=args.timesteps,
        out_dir=RESULTS_DIR,
        final_model_dir=TRAINED_MODEL_DIR,
        n_folds=args.n_folds,
        tb_log_root=RESULTS_DIR / "tb_logs",
    )

    print("\n=== WALK-FORWARD SUMMARY (per-fold val metrics) ===")
    if not metrics_df.empty:
        cols = ["fold", "agent", "val_start", "val_end",
                "total_return", "sharpe", "max_drawdown"]
        print(metrics_df[cols].to_string(index=False))

    print(f"\nFinal-fold models saved to: {TRAINED_MODEL_DIR}")
    print("These can be loaded by tw50_capweighted_finrl_des_backtest.py "
          "to run the 2024-01-02 .. 2026-03-31 test.")


if __name__ == "__main__":
    main()
