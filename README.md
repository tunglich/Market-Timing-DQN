# TW50 DQN — Public Benchmark

Deep Q-Network (DQN, 1D-CNN) applied to Taiwan Top-50 (TW50, 2023-12-29
constituent list) stocks, trading a single symbol at a time on a fixed
candlestick window. Framework: PyTorch + Gymnasium + [ptan](https://github.com/Shmuma/ptan).

- Input features per bar: `<DES>, <OPEN>, <HIGH>, <LOW>, <CLOSE>` (no volume, no sentiment).
- Actions: `Skip / Buy / Close` (long-only single position, no leverage).
- Commissions (TW retail defaults): **buy 0.10 %** / **sell 0.34 %** (0.14 % broker fee tier + 0.30 % securities transaction tax already baked into sell).
- Windows shipped: `55`, `60`, `65`, `75` bars.
- Validation scheme (new code): **5-fold contiguous walk-forward** on `2005-01 ~ 2023-12`.
- Test window (held out): **`2024-01-02 ~ 2026-03-30`**.

---

## Pipeline

```
                    ┌──────────────────────────────────────┐
                    │  data/<sym>_all_<win>.csv            │  50 syms × 4 windows (55/60/65/75)
                    │  <DATE>,<DES>,<OPEN>,<HIGH>,<LOW>,   │  <DES> = binary directional label
                    │  <CLOSE>                             │  from an external encoder
                    └────────────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  src/walk_forward.py                 │  Trim to date < 2024-01-01
                    │        (Stage 1)                     │  Drop leading zero-OHLC rows
                    │                                      │  Split into 5 contiguous chunks
                    └────────────────────┬─────────────────┘  Emit fold_k_train / fold_k_val CSVs
                                         │
                                         │   for k in 0..4  (repeat 5 times)
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  src/train_dqn.py                    │  StocksEnv (Skip/Buy/Close)
                    │        (Stage 2)                     │  1-D CNN (DQNConv1DLarge)
                    │                                      │  PER + n-step (n=2) Double DQN
                    │                                      │  buy 0.10 % / sell 0.34 %
                    │                                      │  Adaptive ε / β schedule
                    └────────────────────┬─────────────────┘  Best val ckpt per fold
                                         │
                                         │   pick best-val ckpt  (or use trained_models/<sym>_all_<win>.data)
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  src/backtest.py                     │  Held-out test
                    │        (Stage 3)                     │  2024-01-02 ~ 2026-03-30
                    │                                      │  Deterministic ε=0 greedy rollout
                    │                                      │  Reports model_pct, bh_pct,
                    │                                      │  excess_pct, n_buy / n_close
                    └────────────────────┬─────────────────┘  → backtest_summary.csv (--all)
                                         │
                                         ▼
                              per-symbol metrics
                              (model_pct vs BH,
                               50 syms × 4 windows)
```

Stage 1 runs once per `(symbol, window)` and produces the 5 fold CSVs.
Stage 2 loops over the 5 folds, saving one `best_val-<reward>.data` per fold under
`saves/<sym>_all_<win>/fold_<k>/`. Stage 3 loads a single checkpoint (either a
shipped `trained_models/` file or a freshly trained one) and runs one
deterministic rollout over the 2024-2026 test window.

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
│   └── backtest.py             # Deterministic test 2024-01-02 ~ 2026-03-30
├── scripts/
│   ├── export_top50_data.py    # Regenerates data/ from a private OHLCV source
│   ├── import_legacy_models.py # Copies best pre-existing TW50 checkpoints
│   └── import_dow30_models.py  # Same, but for the Dow30 universe
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

| Universe | Constituent list | CSVs | Shipped checkpoints | Windows shipped |
|---|---|---|---|---|
| `tw50` (default) | `data/tw50_2023-12-29.csv` (50 syms) | 200 | 89 | 55, 60, 65, 75 (partial) |
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

### 1. Train one symbol × window with 5-fold walk-forward

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
| `--commission-buy` / `--commission-sell` | `0.10` / `0.34` | TW retail defaults (%) |
| `--data-dir` | `data/` | CSV directory to load from |
| `--saves-root` | `saves/` | output root for fold checkpoints and TB logs |
| `--cpu` | off | force CPU even if CUDA is available |

### 2. Backtest on the held-out test window (2024-01-02 ~ 2026-03-30)

```powershell
# single symbol × window against the shipped checkpoint in trained_models/
python src/backtest.py --symbol 2330 --window 55

# override the checkpoint (e.g. a freshly trained fold)
python src/backtest.py --symbol 2330 --window 75 --model saves\2330_all_75\fold_0\best_val-64.770.data

# batch over every shipped TW50 checkpoint (89 of 200 sym × window pairs)
python src/backtest.py --all --out backtest_summary.csv

# batch over the Dow30 universe (90 of 120 sym × window pairs; window 55 not shipped)
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

### 3. Rebuild the price data (optional, private source required)

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
| `python src/backtest.py --universe dow30 --all --out backtest_summary_dow30.csv --cpu` | 90 rows written, `Mean model_pct=+64.40%  Mean bh_pct=+41.25%  Mean excess=+23.15%` (window 60/65/75 excess = +2.45 / +18.48 / +48.51) |
| `python src/train_dqn.py --symbol 2330 --window 75 --fold 0 --hours 0.02 --n-envs 1 --cpu` | best val reward reaches ~65 after ~3.8k frames; checkpoint saved under `saves/2330_all_75/fold_0/` |

These are diagnostic upper-bounds (see the next section), not the final
paper numbers.

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

Coverage note: only **89 of the 200** (sym × window) legacy checkpoints exist
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

