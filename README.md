# TW50 DQN — Public Benchmark

Deep Q-Network (DQN, 1D-CNN) applied to Taiwan Top-50 (TW50, 2023-12-29
constituent list) stocks, trading a single symbol at a time under a
preset directional-signal accuracy.
Framework: PyTorch + Gymnasium + [ptan](https://github.com/Shmuma/ptan).

- Input features per bar: `<DES>, <OPEN>, <HIGH>, <LOW>, <CLOSE>` (no volume, no sentiment).
- Actions: `Skip / Buy / Close` (long-only single position, no leverage).
- Commissions (TW retail, §3.3): **buy 0.1425 %** / **sell 0.4425 %** (0.1425 % broker fee + 0.30 % securities transaction tax).
- Accuracies shipped: `55 %`, `60 %`, `65 %`, `75 %` (target directional accuracy `ρ` of the `<DES>` signal).
- Validation scheme (new code): **5-fold contiguous walk-forward** on `2005-01 ~ 2023-12`.
- Test window (held out): **`2024-01-02 ~ 2026-03-30`**.

> **Naming convention (legacy).** Throughout the code and CLI, the numbers
> `55 / 60 / 65 / 75` are exposed as `--window` and appear in filenames as
> `<sym>_all_<win>.csv` and `saves/<sym>_all_<win>/`. This is a legacy
> holdover; the current interpretation is the **target directional accuracy
> `ρ` (in %)** of the `<DES>` signal, matching the accuracy grid in the paper.
> Existing users can keep passing `--window 75` as before; new prose in this
> README uses `ρ` / "accuracy" for clarity.

---

## Pipeline

```
                    ┌──────────────────────────────────────┐
                    │  Raw OHLCV                           │  TW50: data/{Open,High,Low,Close,
                    │                                      │       Volume}.csv (wide matrix)
                    │                                      │  Dow30: yfinance auto_adjust=True
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  scripts/gen_des_by_accuracy*.py     │  y[i]=1 iff mean(close[i+1..i+h])
                    │        (Stage 1: make features)      │        >= close[i]      (h∈{5,10,20,60})
                    │  --accuracy ρ  --horizon h  --seed   │  DES = y w.p. ρ/100, else 1-y
                    │  (or gen_des_clustered.py, Eq. 5)    │  Per-row independent flip (Eq. 2)
                    └────────────────────┬─────────────────┘  → data/<sym>_all_<ρ>.csv
                                         │                    <DATE>,<DES>,<OPEN>,<HIGH>,
                                         │                    <LOW>,<CLOSE>[,<VOLUME>]
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  src/walk_forward.py                 │  Trim to date < 2024-01-01
                    │        (Stage 2)                     │  Drop leading zero-OHLC rows
                    │                                      │  Split into 5 contiguous chunks
                    └────────────────────┬─────────────────┘  Emit fold_k_train / fold_k_val CSVs
                                         │
                                         │   for k in 0..4  (repeat 5 times)
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  src/train_dqn.py                    │  StocksEnv (Skip/Buy/Close)
                    │        (Stage 3)                     │  1-D CNN (DQNConv1DLarge)
                    │                                      │  PER + n-step (n=2) Double DQN
                    │                                      │  buy 0.1425 % / sell 0.4425 %
                    │                                      │  Adaptive ε / β schedule
                    └────────────────────┬─────────────────┘  Best val ckpt per fold
                                         │
                                         │   pick best-val ckpt  (or use trained_models/<sym>_all_<ρ>.data)
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  src/backtest.py                     │  Held-out test
                    │        (Stage 4)                     │  2024-01-02 ~ 2026-03-30
                    │                                      │  Deterministic ε=0 greedy rollout
                    │                                      │  Reports model_pct, bh_pct,
                    │                                      │  excess_pct, n_buy / n_close
                    └────────────────────┬─────────────────┘  → backtest_summary.csv (--all)
                                         │
                                         ▼
                              per-symbol metrics
                              (model_pct vs BH,
                               50 syms × 4 accuracies)
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  src/portfolio_backtest.py           │  Aggregation
                    │        (Stage 5)                     │  equal / cap (TW-50)
                    │                                      │  equal / price (Dow-30)
                    │                                      │  → portfolio_summary.csv
                    │                                      │  → portfolio_timeseries.csv
                    └──────────────────────────────────────┘
```

**Stage 1 (make features)** takes raw OHLCV plus two knobs — target accuracy
`ρ ∈ [50, 100]` and forecast horizon `h ∈ {5, 10, 20, 60}` — and emits one
`data/<sym>_all_<ρ>.csv` per symbol whose `<DES>` column agrees with the
ground-truth `h`-day directional label with probability `ρ/100`. `h=20` and
`ρ ∈ {55, 60, 65, 75}` reproduce the shipped CSVs.

Stage 2 runs once per `(symbol, accuracy)` and produces the 5 fold CSVs.
Stage 3 loops over the 5 folds, saving one `best_val-<reward>.data` per fold
under `saves/<sym>_all_<ρ>/fold_<k>/`. Stage 4 loads a single checkpoint
(either a shipped `trained_models/` file or a freshly trained one) and runs one
deterministic rollout over the 2024-2026 test window. Stage 5 replays every
shipped checkpoint and aggregates the per-symbol P&L streams into a single
portfolio equity curve under equal / cap (TW-50) or equal / price (Dow-30)
weighting.

---

## Layout

```
public_tw_dqn/
├── data/                       # TW50: 200 CSVs + tw50_2023-12-29.csv
│                               # Dow30: 120 CSVs + dow30_constituents.csv
│   └── <sym>_all_<win>.csv     # <DATE>,<DES>,<OPEN>,<HIGH>,<LOW>,<CLOSE>
├── lib/                        # Model / env / data helpers (verbatim from private repo)
│   ├── data.py                 # CSV reader (auto-handles missing <VOLUME>)
│   ├── environ.py              # StocksEnv (Skip/Buy/Close) with TW-realistic costs
│   ├── models.py               # DQNConv1DLarge (used here) + SimpleFFDQN
│   ├── common.py               # RewardTracker, EpsilonTracker, unpack_batch
│   └── validation.py           # validation_run for held-out reward
├── src/
│   ├── walk_forward.py         # 5-fold contiguous CV over pre-test history
│   ├── train_dqn.py            # PER + n-step DQN trainer (CNN, novol, TW costs)
│   ├── backtest.py             # Deterministic per-symbol test 2024-01-02 ~ 2026-03-30
│   └── portfolio_backtest.py   # Aggregates DQN signals into equal/cap/price portfolio
├── scripts/
│   ├── export_top50_data.py    # Regenerates data/ from a private OHLCV source
│   ├── import_legacy_models.py # Copies best pre-existing TW50 checkpoints
│   ├── import_dow30_models.py  # Same, but for the Dow30 universe
│   ├── gen_des_by_accuracy.py  # Independent-flip DES generator (Eq. 2)
│   ├── gen_des_clustered.py    # Clustered-error DES generator (Eq. 5)
│   ├── accuracy_to_metrics.py  # Directional accuracy -> F1 / AUC (Eq. 6)
│   └── break_even_cost.py      # Extra bp/side that would erase the excess return
├── data/
│   └── dow30_price_weights_2023-12-29.csv  # Dow-30 price-weight reference prices
├── trained_models/             # 89 TW50 + 90 Dow30 checkpoints (LFS)
│                               # + manifest.csv, manifest_dow30.csv
├── requirements.txt
├── LICENSE  (MIT)
├── .gitattributes  (LFS: trained_models/**/*.data)
└── .gitignore
```

## Universes

Two constituent universes ship with the repository. All code paths
(`train_dqn.py`, `backtest.py`, the CNN in `lib/models.py`) are identical
across universes — only the CSV lists and the shipped checkpoints differ.

| Universe | Constituent list | CSVs | Shipped checkpoints | Accuracies shipped |
|---|---|---|---|---|
| `tw50` (default) | `data/tw50_2023-12-29.csv` (50 syms) | 200 | 89 | 55 %, 60 %, 65 %, 75 % (partial) |
| `dow30` | `data/dow30_constituents.csv` (30 syms, post-Nov-2024 lineup) | 120 | 90 | 60, 65, 75 (55 not shipped) |

Select the universe for batch backtests with `--universe`:

```powershell
python src/backtest.py --universe tw50  --all --out backtest_summary.csv
python src/backtest.py --universe dow30 --all --out backtest_summary_dow30.csv
```

Single-symbol runs (`--symbol AAPL --window 75`) auto-detect the CSV; no flag
needed.

## Install

Requires Python 3.10+ (verified on 3.11.15 / Windows 11 + conda).

### Option A — venv (works on Windows / Linux / macOS)

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell
# source .venv/bin/activate         # bash / zsh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda (recommended if you already have miniconda/anaconda)

```powershell
conda create -n mtdqn python=3.11 -y
conda activate mtdqn
pip install -r requirements.txt
```

### GPU (optional)

`requirements.txt` pins the CPU wheels of PyTorch. For CUDA acceleration,
install the matching build **before** `pip install -r requirements.txt` (or
replace `torch` afterwards):

```powershell
# example: CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

Verify the environment before running anything:

```powershell
python -c "import numpy, pandas, torch, gymnasium, ptan, tensorboardX; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
```

## Quick smoke test (≈ 3 minutes, CPU only)

Runs the three moving parts end-to-end so you know your environment is
healthy before committing to a full training run.

```powershell
# from the repo root
# 1) confirm the walk-forward splitter reads the CSVs correctly
python src/walk_forward.py --symbol 2330 --window 75 --dry-run

# 2) backtest one shipped checkpoint (must exist in trained_models/)
python src/backtest.py --symbol 2330 --window 55 --cpu

# 3) train fold 0 for ~72 seconds and save a fresh checkpoint
python src/train_dqn.py --symbol 2330 --window 75 --fold 0 --hours 0.02 --n-envs 1 --cpu

# 4) backtest the freshly trained checkpoint
$ckpt = Get-ChildItem saves\2330_all_75\fold_0\best_val-*.data | Select-Object -Last 1
python src/backtest.py --symbol 2330 --window 75 --model $ckpt.FullName --cpu
```

Expected observations:

- Step 1 prints 5 folds of ~936 val rows each.
- Step 2 prints a line like `2330_55: model=+86.42%  BH=+206.37%  excess=-119.94%`.
- Step 3 ends with `Reached train_hours=0.02h at frame ~3,800` and writes
  `saves/2330_all_75/fold_0/best_val-*.data`.
- Step 4 reports a `model_pct` for the just-trained checkpoint.

## Usage (full workflow)

### 1. Train one symbol × accuracy with 5-fold walk-forward

```powershell
# all 5 folds, 1.5 h wall clock per fold (default)
python src/train_dqn.py --symbol 2330 --window 75 --fold all --hours 1.5

# a specific fold, forced onto CPU, single env
python src/train_dqn.py --symbol 2330 --window 75 --fold 2 --hours 1.5 --n-envs 1 --cpu

# subset of folds
python src/train_dqn.py --symbol 2330 --window 75 --fold 0,2,4 --hours 1.0
```

Fold layout (per symbol, computed after dropping leading zero-OHLC rows):

- Split `date < 2024-01-01` into 5 equal contiguous chunks.
- Fold `k`: validation = chunk `k`; training = concat of the other 4 chunks.

Best-per-fold checkpoints are written under
`saves/<sym>_all_<win>/fold_<k>/best_val-<reward>.data` and TensorBoard event
files under `.../fold_<k>/tb/`. Launch TensorBoard with:

```powershell
tensorboard --logdir saves\2330_all_75
```

To dry-run only the split (no training):

```powershell
python src/walk_forward.py --symbol 2330 --window 75 --dry-run
```

Useful `train_dqn.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--hours` | `1.5` | wall-clock budget per fold |
| `--n-envs` | `4` | parallel envs; drop to `1` on CPU-only boxes |
| `--batch-size` | `128` | mini-batch for PER sampling |
| `--bars` | `10` | context length fed to the 1-D CNN |
| `--lr` / `--gamma` | `1e-4` / `0.95` | Adam LR / discount |
| `--commission-buy` / `--commission-sell` | `0.1425` / `0.4425` | TW retail defaults per §3.3 (%) |
| `--data-dir` | `data/` | CSV directory to load from |
| `--saves-root` | `saves/` | output root for fold checkpoints and TB logs |
| `--cpu` | off | force CPU even if CUDA is available |

### 2. Backtest on the held-out test window (2024-01-02 ~ 2026-03-30)

```powershell
# single symbol × accuracy against the shipped checkpoint in trained_models/
python src/backtest.py --symbol 2330 --window 55

# override the checkpoint (e.g. a freshly trained fold)
python src/backtest.py --symbol 2330 --window 75 --model saves\2330_all_75\fold_0\best_val-64.770.data

# batch over every shipped TW50 checkpoint (89 of 200 sym × accuracy pairs)
python src/backtest.py --all --out backtest_summary.csv

# batch over the Dow30 universe (90 of 120 sym × accuracy pairs; accuracy 55 % not shipped)
python src/backtest.py --universe dow30 --all --out backtest_summary_dow30.csv

# single Dow30 symbol
python src/backtest.py --symbol AAPL --window 75
```

The batch command writes `backtest_summary.csv` with columns
`symbol,window,test_start,test_end,n_rows,model_pct,bh_pct,excess_pct,n_buy,n_close,n_skip,model_file`
and prints a `Mean model_pct=... Mean bh_pct=... Mean excess=...` footer.

Useful `backtest.py` flags: `--commission-buy`, `--commission-sell`,
`--bars`, `--epsilon` (defaults to 0.0 for a deterministic run), `--cpu`,
`--models-dir`, `--data-dir`.

### 3. Portfolio backtest (aggregate the per-symbol DQN signals)

`src/portfolio_backtest.py` replays every shipped `<sym>_all_<W>.data`
checkpoint over the same 2024-01-02 ~ 2026-03-30 test window, captures the
environment's per-step P&L, and aggregates it into a single TW-50 portfolio
equity curve. Two weighting variants are computed per accuracy:

* **equal** — `1/N` over every covered symbol.
* **cap**   — TW-50 market-cap weights from `data/tw50_2023-12-29.csv`,
              renormalised over the covered symbols.
* **price** — Dow-30 price weights (Dow Jones Industrial Average convention)
              built from the 2023-12-29 close of every constituent, shipped
              as `data/dow30_price_weights_2023-12-29.csv` and renormalised
              over the covered symbols.

```powershell
# TW-50: all 4 accuracies, both schemes (equal + cap)
python src/portfolio_backtest.py --cpu

# TW-50 accuracy 65 % + 75 %, equal only
python src/portfolio_backtest.py --windows 65 75 --scheme equal --cpu

# Dow-30: all 4 accuracies, both schemes (equal + price)
python src/portfolio_backtest.py --universe dow30 --cpu `
    --out-summary portfolio_summary_dow30.csv `
    --out-timeseries portfolio_timeseries_dow30.csv
```

Outputs (repo root by default):

* `portfolio_summary.csv` — one row per (window, scheme) with
  `dqn_final_pct / dqn_sharpe / dqn_max_dd_pct / bh_final_pct / bh_sharpe /
  bh_max_dd_pct / excess_pct`.
* `portfolio_timeseries.csv` — daily equity curves for every
  `dqn_<univ>_w<W>_<scheme>` and `bh_<univ>_w<W>_<scheme>` column.

Cap-weight is defined for `--universe tw50`; price-weight is defined for
`--universe dow30`. Equal-weight works for both. `--scheme both` expands to
`(equal, cap)` on TW-50 and `(equal, price)` on Dow-30.

### 4. Rebuild the price data (optional, private source required)

```powershell
python scripts/export_top50_data.py --overwrite
```

This regenerates every `data/<sym>_all_<w>.csv` from a private OHLCV source at
`d:/DRL/data/` (dropping the `<VOLUME>` column). External users can point
`--source` at their own directory that produces the same input schema. The
CSVs already shipped in `data/` are sufficient for training and backtesting —
you only need this step if you are extending the universe or refreshing the
data.

## Verified smoke-test numbers (2026-07 build, CPU)

| Command | Result |
|---|---|
| `python src/backtest.py --symbol 2330 --window 55 --cpu` | `model=+86.42%  BH=+206.37%  excess=-119.94%  rows=539  buys=267 closes=68` |
| `python src/backtest.py --all --out backtest_summary.csv --cpu` | 89 rows written, `Mean model_pct=+59.20%  Mean bh_pct=+41.79%  Mean excess=+17.41%` |
| `python src/backtest.py --symbol AAPL --window 75 --cpu` | `model=+80.62%  BH=+32.71%  excess=+47.91%  rows=562  buys=135 closes=229` |
| `python src/backtest.py --universe dow30 --all --out backtest_summary_dow30.csv --cpu` | 90 rows written, `Mean model_pct=+64.40%  Mean bh_pct=+41.25%  Mean excess=+23.15%` (accuracy 60/65/75 % excess = +2.45 / +18.48 / +48.51) |
| `python src/train_dqn.py --symbol 2330 --window 75 --fold 0 --hours 0.02 --n-envs 1 --cpu` | best val reward reaches ~65 after ~3.8k frames; checkpoint saved under `saves/2330_all_75/fold_0/` |

These are diagnostic upper-bounds (see the next section), not the final
paper numbers.

### TW-50 portfolio numbers (`src/portfolio_backtest.py`, 2024-01-02 ~ 2026-03-30, TW retail costs 0.1425 / 0.4425% per §3.3)

| window | scheme | n | DQN final % | DQN Sharpe | DQN maxDD % | BH final % | BH Sharpe | BH maxDD % | excess % |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 55 | equal | 25 | +34.06  | 1.10 |  –11.86 | +40.48  | 0.98 | –23.47 |  –6.41 |
| 55 | cap   | 25 | +75.54  | 1.28 |  –22.22 | +126.89 | 1.68 | –25.32 | –51.35 |
| 60 | equal | 25 | +48.92  | 1.60 |   –8.64 | +49.65  | 1.14 | –23.26 |  –0.74 |
| 60 | cap   | 25 | +46.35  | 1.44 |   –8.90 | +44.47  | 1.02 | –25.68 |  +1.87 |
| 65 | equal | 25 | +81.59  | 2.62 |   –6.63 | +49.65  | 1.14 | –23.26 | +31.94 |
| 65 | cap   | 25 | +79.11  | 2.49 |   –6.54 | +44.47  | 1.02 | –25.68 | +34.64 |
| 75 | equal | 14 | +77.08  | 3.15 |   –4.36 | +31.69  | 0.94 | –16.48 | +45.39 |
| 75 | cap   | 14 | +81.23  | 3.02 |   –5.88 | +26.14  | 0.75 | –22.26 | +55.09 |

*Returns estimated under paper §3.3 costs (buy 0.1425% / sell 0.4425%). Original values were computed under 0.10%/0.34%; adjusted by subtracting per-stock average cost drag (Δbuy = 0.0425 pp × n\_buy + Δsell = 0.1025 pp × n\_close, averaged across covered symbols). Checkpoints are stored as Git LFS objects; run `git lfs pull` then re-run `src/portfolio_backtest.py` to reproduce exactly.*

Sharpe is annualised on 252 trading days; final % is compounded. `n` is the
number of TW-50 symbols with a shipped checkpoint at that accuracy (see
[trained_models/manifest.csv](trained_models/manifest.csv) for missing rows).

**Comparison with the FinRL SB3 variants** (from
[finrl/results/capweighted_finrl_des75/summary.txt](finrl/results/capweighted_finrl_des75/summary.txt),
covering 50 stocks with a portfolio-allocation agent instead of per-symbol
timing, `2024-01-02 ~ 2026-03-31`, no transaction cost modelled):

| Agent | final % | Sharpe | maxDD % |
|---|---:|---:|---:|
| A2C            | +143.42 | 1.66 | –26.47 |
| PPO            | +101.74 | 1.51 | –25.78 |
| DDPG           | +119.19 | 1.54 | –26.62 |
| TD3            |  +37.17 | 0.86 | –23.22 |
| SAC            |  +77.03 | 1.21 | –30.16 |
| CAP\_BH (50)   | +109.16 | 1.48 | –26.79 |
| TWII           |  +82.14 | 1.33 | –28.69 |
| **DQN cap w=75** (this repo, 14 syms) | **+81.23** | **3.02** | **–5.88** |
| **DQN eq w=75** (this repo, 14 syms)  | **+77.08** | **3.15** | **–4.36** |

The DQN portfolio's edge is on the risk axis (Sharpe 3.0–3.2 vs ≤ 1.7 for the
SB3 variants and CAP\_BH; maxDD kept under 6% vs 23–30%). Absolute return is
lower than A2C partly because DQN w=75 covers only 14 stocks vs 50.

### Dow-30 portfolio numbers (`src/portfolio_backtest.py --universe dow30`, 2024-01-02 ~ 2026-03-30, US retail costs 0.05 / 0.05%)

| window | scheme | n | DQN final % | DQN Sharpe | DQN maxDD % | BH final % | BH Sharpe | BH maxDD % | excess % |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | equal | 30 |  +53.49 | 2.28 |  –5.97 | +39.05 | 1.18 | –15.41 | +14.44 |
| 60 | price | 30 |  +47.42 | 2.00 |  –6.23 | +26.67 | 0.83 | –15.01 | +20.76 |
| 65 | equal | 30 |  +80.50 | 3.55 |  –3.35 | +39.05 | 1.18 | –15.41 | +41.45 |
| 65 | price | 30 |  +77.43 | 3.27 |  –4.25 | +26.67 | 0.83 | –15.01 | +50.76 |
| 75 | equal | 30 | +143.85 | 5.75 |  –2.10 | +39.05 | 1.18 | –15.41 | +104.79 |
| 75 | price | 30 | +138.50 | 5.45 |  –2.14 | +26.67 | 0.83 | –15.01 | +111.83 |

Accuracy 55 % is skipped (no Dow-30 checkpoints at that accuracy). The
price-weighted BH benchmark is materially harder to beat because a handful of
high-price constituents (SHW, UNH, MSFT, GS) drag the DJIA-style index; the
DQN portfolio still delivers +50 to +112 pp of excess return over it.

## Paper-driven add-ons

Utilities that reproduce the methodology-specific numbers from the SC-DQN
paper without needing to retrain any checkpoint.

**Clustered directional-error DES (Eq. 5).** Replaces the independent-flip
label noise model (`scripts/gen_des_by_accuracy.py`, Eq. 2) with an AR(1)
idiosyncratic component plus a shared cross-sectional stress factor, so that
mislabels concentrate in high-volatility regimes:

```powershell
python scripts/gen_des_clustered.py --universe tw50 --accuracy 65 `
    --beta 0.5 --phi 0.3 --seed 42
python scripts/gen_des_clustered.py --universe dow30 --accuracy 65 `
    --beta 0.5 --phi 0.3 --seed 42
```

`beta=phi=0` recovers the independent baseline. Output goes to
`data/clustered_b<BB>_p<PP>/<sym>_all_<acc>.csv` with the same schema as the
existing DES files, so downstream training is unchanged.

**Directional accuracy → F1 / AUC (Eq. 6).** Closed-form mapping that lets a
practitioner screen a candidate binary classifier against a target trading
edge before running a full backtest:

```powershell
python scripts/accuracy_to_metrics.py --accuracy 0.65 --base-rate 0.53
# accuracy rho = 0.650   AUC = 0.650   F1 = 0.663   (matches paper Table 2)

python scripts/accuracy_to_metrics.py --grid --out reports/accuracy_map.csv
```

**Break-even round-trip friction (Table 3).** Given any
`backtest_summary*.csv` and its matching `portfolio_summary*.csv`, reports
the extra bp per side that would erase the portfolio's excess return over
buy-and-hold:

```powershell
python scripts/break_even_cost.py `
    --backtest-summary backtest_summary_dow30.csv `
    --portfolio-summary portfolio_summary_dow30.csv `
    --out reports/break_even_dow30.csv
```

Verified output (2026-07 build):

| universe | window | scheme | excess bp | sides/stock | break-even bp/side |
|---|---:|:---:|---:|---:|---:|
| dow30 | 60 | equal |  1443.7 | 340.5 |   4.24 |
| dow30 | 65 | equal |  4144.6 | 344.2 |  12.04 |
| dow30 | 65 | price |  5075.9 | 344.2 |  14.75 |
| dow30 | 75 | equal | 10479.3 | 329.4 |  31.81 |
| dow30 | 75 | price | 11183.2 | 329.4 |  33.95 |

## Important: code vs shipped checkpoints

The `trained_models/*.data` files in this repo were **trained under an older
scheme**:

- **Legacy training** (produced the shipped checkpoints): fixed split, train =
  `2005-01-03 ~ 2023-12-29`, validation = `2024-01-02 ~ 2026-03-31`.
- **New training** (this repo's `src/train_dqn.py`): 5-fold contiguous
  walk-forward on `2005-01 ~ 2023-12`, with `2024-01-02 ~ 2026-03-30` held out
  as a completely unseen test set.

The shipped checkpoints therefore **peeked at the walk-forward test window
during their own validation**. Treat their `src/backtest.py` numbers as an
upper-bound diagnostic, not a clean generalisation estimate. To reproduce
clean numbers, retrain with `src/train_dqn.py` and then run `src/backtest.py`
against the newly saved checkpoints.

Coverage note: only **89 of the 200** (sym × accuracy) legacy checkpoints exist
in the source repo. Missing pairs are listed in
`trained_models/manifest.csv` with `notes = "MISSING legacy checkpoint"`.

## Data schema

| Column      | Meaning                                              |
|-------------|------------------------------------------------------|
| `<DATE>`    | YYYY-MM-DD trading day. Ignored by `lib.data`.       |
| `<DES>`     | Binary label from the internal encoder ({0,1}).      |
| `<OPEN>`    | Session open price (TWD).                            |
| `<HIGH>`    | Session high (TWD).                                  |
| `<LOW>`     | Session low (TWD).                                   |
| `<CLOSE>`   | Session close (TWD).                                 |

`lib.data.load_relative` normalises OHLC to open-relative returns and stacks
them with `<DES>` along the channel axis before the CNN.

## Trained-model manifest

`trained_models/manifest.csv` records, per (symbol, window):

- `val_reward`: the reward embedded in the legacy filename.
- `source_folder`, `source_ckpt`: origin under `d:/DRL/saves/`.
- `notes`: `MISSING legacy checkpoint`, `exists (kept)`, etc.

## License

MIT — see [LICENSE](LICENSE).

