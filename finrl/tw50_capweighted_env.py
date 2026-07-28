"""Cap-anchored portfolio env: agent learns deviation from cap-weights.

Differs from FinRL's StockPortfolioEnv:

1. Action space is Box(-3, 3) instead of (0, 1).
2. Weights = softmax(log(cap_weights) + action). When action=0, weights = cap_w.
   Agent's policy thus expresses "log-odds deviation" from cap baseline.
3. Optional reward penalty: subtracts lambda_dev * sum((w - cap_w)^2) * value
   from reward to discourage extreme deviations during training.

Tracks `deviation_memory` for diagnostics.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


class CapAnchoredPortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        cap_weights,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        lambda_dev=0.0,
        action_clip=3.0,
        day=0,
    ):
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list
        self.lambda_dev = float(lambda_dev)

        self.action_space = spaces.Box(low=-action_clip, high=action_clip,
                                       shape=(action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_space + len(tech_indicator_list), state_space),
        )

        self.data = self.df.loc[self.day, :]
        ordered_tics = list(self.data.tic.values)

        cap = np.array([float(cap_weights.get(t, 0.0)) for t in ordered_tics])
        cap = np.maximum(cap, 1e-8)
        cap = cap / cap.sum()
        self.cap_weights_arr = cap
        self.cap_logits = np.log(cap)

        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[t].values.tolist() for t in tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount

        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0.0]
        self.actions_memory = [list(self.cap_weights_arr)]
        self.deviation_memory = [0.0]
        self.date_memory = [self.data.date.unique()[0]]
        self.reward = 0.0

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")
            df_dr = pd.DataFrame(self.portfolio_return_memory, columns=["daily_return"])
            if df_dr["daily_return"].std() != 0:
                sharpe = (252 ** 0.5) * df_dr["daily_return"].mean() / df_dr["daily_return"].std()
                print(f"Sharpe: {sharpe:.4f}")
            avg_dev = float(np.mean(self.deviation_memory))
            print(f"Avg deviation from cap: {avg_dev:.5f}")
            print("=================================")
            return self.state, self.reward, self.terminal, False, {}

        actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        logits = self.cap_logits + actions
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        weights = exp / np.sum(exp)
        deviation = float(np.sum((weights - self.cap_weights_arr) ** 2))

        self.actions_memory.append(weights)
        self.deviation_memory.append(deviation)
        last_day = self.data

        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[t].values.tolist() for t in self.tech_indicator_list],
            axis=0,
        )

        portfolio_return = float(np.sum(
            ((self.data.close.values / last_day.close.values) - 1.0) * weights
        ))
        new_portfolio_value = self.portfolio_value * (1.0 + portfolio_return)
        self.portfolio_value = new_portfolio_value

        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.data.date.unique()[0])
        self.asset_memory.append(new_portfolio_value)

        if self.lambda_dev > 0:
            self.reward = new_portfolio_value * (1.0 - self.lambda_dev * deviation)
        else:
            self.reward = new_portfolio_value

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[t].values.tolist() for t in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0.0]
        self.actions_memory = [list(self.cap_weights_arr)]
        self.deviation_memory = [0.0]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state, {}

    def render(self, mode="human"):
        return self.state

    def save_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory,
                             "daily_return": self.portfolio_return_memory})

    def save_action_memory(self):
        df_date = pd.DataFrame(self.date_memory, columns=["date"])
        df_actions = pd.DataFrame(self.actions_memory, columns=self.data.tic.values)
        df_actions.index = df_date.date
        return df_actions

    def save_deviation_memory(self):
        df_date = pd.DataFrame(self.date_memory, columns=["date"])
        df_dev = pd.DataFrame({"deviation": self.deviation_memory}, index=df_date.date)
        return df_dev

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
