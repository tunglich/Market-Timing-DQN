# TW-50 FinRL Portfolio Backtest (A2C · PPO · DDPG · TD3 · SAC)

This folder contains a **FinRL-based portfolio-allocation experiment** on the
Taiwan Top-50 (TW-50) universe, comparing five stable-baselines3 agents
(A2C, PPO, DDPG, TD3, SAC) under a **cap-weight-anchored** custom environment.
Five variants are included:

| Variant | Prep script | Feature set | Notes |
|---|---|---|---|
| `baseline` | `prep_data/tw50_2024_20260331_prep_data.py` | 8 FinRL indicators | Vanilla FinRL `StockTradingEnv` |
| `capweighted_finrl` | `prep_data/tw50_capweighted_finrl_prep.py` | 8 FinRL indicators + 252-day cov | `CapAnchoredPortfolioEnv` (softmax around TW-50 cap-weights) |
| `capweighted_finrl_des75` | `prep_data/tw50_capweighted_finrl_des_prep.py` | 8 indicators + **DES @ 75% accuracy** + cov | DES = Direction of Expected Sign, oracle-noise label |
| `capweighted_finrl_des65` | `prep_data/tw50_capweighted_finrl_des65_prep.py` | 8 indicators + **DES @ 65%** + cov | |
| `capweighted_finrl_des60` | `prep_data/tw50_capweighted_finrl_des60_prep.py` | 8 indicators + **DES @ 60%** + cov | |

## Experiment design

- **In-sample period:** 2005-01-03 → 2023-12-29
- **Out-of-sample test period:** 2024-01-02 → 2026-03-31
- **Universe:** TW-50 constituents as of 2023-12-29 (list in `../data/tw50_2023-12-29.csv`)
- **Weights:** `w = softmax(log(cap_w) + action)` with `action ∈ [-3, +3]`,
  reward shaping: `pv * (1 - lambda_dev * Σ(w - cap_w)^2)`, `lambda_dev = 0.5`
- **Training budget:** 60,000 timesteps per agent (single-shot); walk-forward
  splits this into 4 folds of 15,000 timesteps each by default.
- **Walk-forward validation:** 4-fold expanding window over the 5 chronological
  segments of the in-sample period. The **model produced by fold 4** (trained
  on S1..S4, validated on S5) is the one used for the 2024-2026 test — no
  refit on the full sample.

  ```
  S1        S2        S3        S4        S5        │  TEST
  2005..    ....      ....      ....      ..2023-12 │  2024..2026-03
  ├───────► val
        ├──────────► val
              ├──────────────► val
                    ├──────────────────► val
                                              ┌──────────►
                                          fold-4 model rolls forward
  ```

## Layout

```
finrl/
├── stock_trading_env_fixed.py       # Patched FinRL StockTradingEnv (baseline)
├── tw50_capweighted_env.py          # CapAnchoredPortfolioEnv (custom)
├── walk_forward.py                  # Reusable walk-forward driver
├── prep_data/                       # 5 scripts producing train/trade pkls
├── train/                           # 5 single-shot trainers + 1 walk-forward driver
├── backtest/                        # 5 backtest scripts (produce metrics + weight csvs)
└── results/                         # Backtest artifacts checked into the repo
    ├── baseline/
    ├── capweighted_finrl/
    ├── capweighted_finrl_des75/     # backtest_results.csv, metrics.csv, summary.txt,
    ├── capweighted_finrl_des65/     # *_weights.csv, backtest_compare.png,
    └── capweighted_finrl_des60/     # weight_diagnostics.csv
```

Data files live under `../data/` in this repo:
- `Open.csv`, `High.csv`, `Low.csv`, `Close.csv`, `Volume.csv` — wide OHLCV matrices
- `<ticker>_all_{55,60,65,75}.csv` — 50 tickers × 4 DES-accuracy labeled files
  (used to inject the noisy DES feature into each variant)
- `tw50_2023-12-29.csv` — TW-50 constituents + cap-weights

## Reproducing

1. **Install FinRL and dependencies.** This subfolder does **not** vendor the
   upstream `finrl` Python package — install it in your environment:

   ```powershell
   pip install finrl
   # or, from source: pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
   ```

   Additional deps: `stable-baselines3`, `yfinance`, `pandas`, `numpy`,
   `matplotlib`, `stockstats` (all pulled in transitively by FinRL).

2. **Prepare training/test pickles.** From `finrl/`:

   ```powershell
   python prep_data/tw50_capweighted_finrl_des_prep.py       # DES @ 75%
   ```

   This writes `tw50_capweighted_finrl_des_train.pkl` and
   `tw50_capweighted_finrl_des_trade.pkl` into `finrl/`. These pickles are
   ~549 MB each and are **not** checked into the repo — regenerate locally.

3. **Train the five agents.** Either single-shot (baseline behaviour):

   ```powershell
   python train/tw50_capweighted_finrl_des_train.py
   ```

   or with 4-fold expanding-window walk-forward:

   ```powershell
   python train/tw50_capweighted_finrl_des_walk_forward.py --timesteps 15000
   ```

   Either mode writes `agent_<name>.zip` into
   `finrl/tw50_capweighted_finrl_des_trained_models/`. The walk-forward driver
   also writes per-fold validation metrics to
   `finrl/tw50_capweighted_finrl_des_walkforward_results/walkforward_metrics.csv`.

4. **Run the 2024-2026 backtest:**

   ```powershell
   python backtest/tw50_capweighted_finrl_des_backtest.py
   ```

   Outputs are written into `results/capweighted_finrl_des75/` (name follows
   the variant, not the script name).

Repeat steps 2–4 for the other four variants by substituting the prep / train
/ backtest script names in the table at the top.

## Notes on this fork

- `stock_trading_env_fixed.py` is a locally patched copy of
  `finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv` with the
  reset seed / action-space fix (see the file header). The baseline backtest
  imports from this file rather than the upstream module.
- `tw50_capweighted_env.py` (custom, no upstream equivalent) parameterises the
  cap-weight anchor `cap_w`, deviation penalty `lambda_dev`, and action clip
  `action_clip`.
- `walk_forward.py` is agent-agnostic: it accepts any FinRL portfolio env
  constructor and any subset of `{a2c, ppo, ddpg, td3, sac}`, so the same
  driver can be reused for the other DES-accuracy variants by writing a thin
  wrapper that mirrors `tw50_capweighted_finrl_des_walk_forward.py`.
