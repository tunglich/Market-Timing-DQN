"""Train 5 SB3 agents (A2C, PPO, DDPG, TD3, SAC) on TW50 train_data.csv.

Uses Taiwan-realistic commissions (0.1425% buy / 0.4425% sell incl. 0.3%
securities transaction tax).

Saves models to tw50_trained_models/agent_<name> and TB logs to
tw50_results/<name>.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

FINRL_ROOT = Path(__file__).resolve().parent.parent          # Market-Timing-DQN/finrl
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))
# Seed all RNGs for Figure 8 reproducibility (override via FINRL_SEED env var).
from finrl_seeds import read_seed_from_env, set_finrl_seeds
SEED = set_finrl_seeds(read_seed_from_env(default=42))


import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

TIMESTEPS = 20000
TRAINED_MODEL_DIR = str(FINRL_ROOT / "tw50_trained_models")
RESULTS_DIR = str(FINRL_ROOT / "tw50_results")
BUY_COST = 0.001425
SELL_COST = 0.004425  # 0.1425% commission + 0.3% securities transaction tax

os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=== Loading tw50_train_data.csv ===")
train = pd.read_csv("tw50_train_data.csv")
train = train.set_index(train.columns[0]); train.index.names = [""]
print(f"train rows={len(train):,}  tickers={train.tic.nunique()}  "
      f"dates={train['date'].min()} ~ {train['date'].max()}")

stock_dim = len(train.tic.unique())
state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
print(f"Stock dim={stock_dim}  State space={state_space}")
print(f"Buy/sell cost: {BUY_COST*100:.4f}% / {SELL_COST*100:.4f}%")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1_000_000,
    "num_stock_shares": [0] * stock_dim,
    "buy_cost_pct": [BUY_COST] * stock_dim,
    "sell_cost_pct": [SELL_COST] * stock_dim,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
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


PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
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
