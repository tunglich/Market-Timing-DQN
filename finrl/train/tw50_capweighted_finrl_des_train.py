"""Train 5 SB3 agents with CapAnchoredPortfolioEnv:
  features = FinRL 8 indicators + DES + 252-day cov
  TIMESTEPS = 60000  (3x of prior TW50 cap-anchored variant's 20K default)
  init weights = TW50 market-cap weights (action=0 => weights=cap_w)
  reward shaping: portfolio_value * (1 - lambda_dev * sum((w - cap_w)^2))
"""
from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from pathlib import Path

FINRL_ROOT = Path(__file__).resolve().parent.parent          # Market-Timing-DQN/finrl
DATA_DIR = FINRL_ROOT.parent / "data"                    # Market-Timing-DQN/data
TW50_LIST = DATA_DIR / "tw50_2023-12-29.csv"
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))
# Seed all RNGs for Figure 8 reproducibility (override via FINRL_SEED env var).
from finrl_seeds import read_seed_from_env, set_finrl_seeds
SEED = set_finrl_seeds(read_seed_from_env(default=42))


import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from tw50_capweighted_env import CapAnchoredPortfolioEnv

TIMESTEPS = 60_000
LAMBDA_DEV = 0.5
ACTION_CLIP = 3.0

TRAINED_MODEL_DIR = str(FINRL_ROOT / "tw50_capweighted_finrl_des_trained_models")
RESULTS_DIR = str(FINRL_ROOT / "tw50_capweighted_finrl_des_results")
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30",
              "cci_30", "dx_30", "close_30_sma", "close_60_sma", "des"]

os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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


print("=== Loading tw50_capweighted_finrl_des_train.pkl ===")
train = pd.read_pickle("tw50_capweighted_finrl_des_train.pkl")
train.index = train.date.factorize()[0]
print(f"train rows={len(train):,}  tickers={train.tic.nunique()}  "
      f"dates={train['date'].min()} ~ {train['date'].max()}")
print(f"Cols: {list(train.columns)}")

cap_w = load_tw50_weights()
print(f"\nLoaded {len(cap_w)} cap-weights from {TW50_LIST.name}")
print("Top 5:", sorted(cap_w.items(), key=lambda x: -x[1])[:5])

stock_dim = train.tic.nunique()
state_space = stock_dim
print(f"Stock dim={stock_dim}  Indicators ({len(INDICATORS)}): {INDICATORS}")
print(f"State shape=({state_space + len(INDICATORS)}, {state_space})")
print(f"lambda_dev={LAMBDA_DEV}  action_clip=+/-{ACTION_CLIP}")
print(f"Timesteps: {TIMESTEPS:,}  (3x of prior TW50 cap-anchored 20K default)")
print("Action: weights = softmax(log(cap_w) + action)  =>  action=0 => weights=cap_w")

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

e_train_gym = CapAnchoredPortfolioEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()


def train_one(name: str, model_kwargs: dict | None = None) -> None:
    print(f"\n{'=' * 70}\n--- Training {name.upper()} ({TIMESTEPS:,} timesteps) ---\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        agent = DRLAgent(env=env_train)
        model = agent.get_model(name, model_kwargs=model_kwargs) if model_kwargs else agent.get_model(name)
        log_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(log_dir, exist_ok=True)
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        trained = agent.train_model(model=model, tb_log_name=name, total_timesteps=TIMESTEPS)
        save_path = os.path.join(TRAINED_MODEL_DIR, f"agent_{name}")
        trained.save(save_path)
        dt = time.time() - t0
        print(f"--- {name.upper()} done in {dt/60:.1f} min, saved to {save_path}.zip ---", flush=True)
    except Exception:
        dt = time.time() - t0
        print(f"--- {name.upper()} FAILED after {dt/60:.1f} min ---", flush=True)
        traceback.print_exc()


PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 128}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {"batch_size": 128, "buffer_size": 100000, "learning_rate": 0.0001,
              "learning_starts": 100, "ent_coef": "auto_0.1"}

train_one("a2c")
train_one("ppo", PPO_PARAMS)
train_one("ddpg")
train_one("td3", TD3_PARAMS)
train_one("sac", SAC_PARAMS)

print("\n=== ALL AGENTS DONE ===")
print("Models in:", TRAINED_MODEL_DIR)
print("TB logs in:", RESULTS_DIR)
