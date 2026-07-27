# Data schema

## Price CSVs (`data/<sym>_all_<window>.csv`)

Header:

```
<DATE>,<DES>,<OPEN>,<HIGH>,<LOW>,<CLOSE>
```

- `<DATE>` — `YYYY-MM-DD`. Ignored by `lib.data.read_csv`; used only by
  `src/walk_forward.py` and `src/backtest.py` when slicing folds and the test
  window.
- `<DES>` — integer ∈ {0, 1}. Directional/context flag from the private
  candlestick encoder; treated as a binary channel by the CNN.
- `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>` — TWD, unadjusted. Rows where any of
  OHLC ≤ 0 are dropped upstream (pre-IPO / suspension days).

Windows: `55, 60, 65, 75` — refer to the encoder's look-back parameter used to
generate `<DES>`.

## Constituent list (`data/tw50_2023-12-29.csv`)

TW50 index constituents snapshotted on 2023-12-29. Used by
`scripts/export_top50_data.py` and `scripts/import_legacy_models.py` to know
which 50 symbols to iterate over.

## Fold / split boundaries

Per-symbol effective start = first row where all of `<OPEN>, <HIGH>, <LOW>,
<CLOSE>` are strictly positive (some TW50 IPOs post-date 2005).

- **Pre-test span** = `first_ok_row ~ 2023-12-31`.
- **5-fold split** = 5 equal contiguous chunks of the pre-test span.
- **Test window** = `2024-01-02 ~ 2026-03-30` (inclusive), used only by
  `src/backtest.py`.

## Ignored artefacts

`saves/`, `runs/`, `signal/`, `best_model/`, `backtest_*/`, `rewards-*.png`
and TensorBoard event files are `.gitignore`-d; regenerate them locally.
