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

## Layout

```
public_tw_dqn/
├── data/                       # 200 CSVs (50 syms × 4 windows) + tw50_2023-12-29.csv
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
│   └── import_legacy_models.py # Copies best pre-existing checkpoints into trained_models/
├── trained_models/             # 89 legacy checkpoints (LFS) + manifest.csv
├── requirements.txt
├── LICENSE  (MIT)
├── .gitattributes  (LFS: trained_models/**/*.data)
└── .gitignore
```

## Install

Requires Python 3.10+ (tested on 3.11).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU training, install the matching CUDA build of PyTorch first.

## Usage

### 1. Train one symbol × window with 5-fold walk-forward

```bash
python src/train_dqn.py --symbol 2330 --window 75 --fold all --hours 1.5
```

Fold layout (per symbol, computed after dropping leading zero-OHLC rows):

- Split `date < 2024-01-01` into 5 equal contiguous chunks.
- Fold `k`: validation = chunk `k`; training = concat of the other 4 chunks.

Best-per-fold checkpoints are written under
`saves/<sym>_all_<win>/fold_<k>/best_val-<reward>.data` and TensorBoard event
files under `.../fold_<k>/tb/`.

To dry-run only the split (no training):

```bash
python src/walk_forward.py --symbol 2330 --window 75 --dry-run
```

### 2. Backtest on the held-out test window

```bash
python src/backtest.py --symbol 2330 --window 75
python src/backtest.py --all --out backtest_summary.csv     # batch over 50×4
```

Reports per-symbol model return %, Buy&Hold return %, and their difference
across `2024-01-02 ~ 2026-03-30`.

### 3. Rebuild the price data (optional)

```bash
python scripts/export_top50_data.py --overwrite
```

This regenerates every `data/<sym>_all_<w>.csv` from a private OHLCV source at
`d:/DRL/data/` (dropping the `<VOLUME>` column). External users can point
`--source` at their own directory that produces the same input schema.

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

