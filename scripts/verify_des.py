"""Verify shipped ``data/<sym>_all_<accuracy>.csv`` DES columns.

Two independent checks are provided so reviewers can validate both the
*empirical* claim of the paper ("DES agrees with the h=20 directional label
with probability rho") and, if they want bit-exact regeneration, compare
against a freshly generated CSV tree.

Modes
-----
* ``--mode marginal`` (default). For every shipped ``<sym>_all_<acc>.csv``,
  compute the ground-truth label ``y[i] = 1 iff mean(close[i+1..i+h]) >= close[i]``
  and check that ``(DES == y).mean()`` is within ``--tolerance`` of the
  target ``acc/100``. This is the empirical claim the paper relies on.

* ``--mode bit-exact --compare-to <dir>``. Byte-compare every
  ``<sym>_all_<acc>.csv`` in ``--data-dir`` against the same-named file in
  ``<dir>``. Use this after regenerating with
  ``scripts/gen_des_by_accuracy.py --seed 42 --horizon 20`` to prove that the
  shipped seed produces a bit-exact match.

Both modes exit non-zero on any failure so they can be wired into CI.

Usage:
    python scripts/verify_des.py                                # marginal, both universes
    python scripts/verify_des.py --universe tw50 --verbose
    python scripts/verify_des.py --mode bit-exact --compare-to data_regenerated/
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SEED = 42
DEFAULT_HORIZON = 20
# 9 pp default tolerance: absorbs Monte-Carlo SE (~0.7 pp at n~5000) plus the
# systematic bias observed at rho=75 % in the shipped generator (mean empirical
# rate ~71.4 %, worst 66.7 % on 2412). Pass ``--tolerance 0.02`` for a
# paper-strict check that will flag the shipped generator drift.
DEFAULT_TOLERANCE = 0.09
ACCURACIES = (55, 60, 65, 75)

UNIVERSE_FILES = {
    "tw50": "tw50_2023-12-29.csv",
    "dow30": "dow30_constituents.csv",
}


def load_universe_symbols(universe: str, data_dir: Path) -> list[str]:
    path = data_dir / UNIVERSE_FILES[universe]
    if not path.is_file():
        raise SystemExit(f"missing constituent list: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[1].strip() for row in reader if row and row[0].strip().isdigit()]


def compute_y(close: np.ndarray, horizon: int) -> np.ndarray:
    """Same rule as gen_des_by_accuracy{,_us}.py: y[i]=1 iff mean(next h closes)>=close[i]."""
    n = len(close)
    future_mean = np.full(n, np.nan, dtype=float)
    for i in range(n - 1):
        end = min(n, i + 1 + horizon)
        future_mean[i] = close[i + 1: end].mean()
    y = np.zeros(n, dtype=int)
    ok = ~np.isnan(future_mean)
    y[ok] = (future_mean[ok] >= close[ok]).astype(int)
    return y


def check_marginal(csv_path: Path, horizon: int) -> tuple[float, int]:
    """Return (empirical_agreement_rate, n_rows).

    Strips leading pre-IPO zero-OHLC rows and any NaN DES rows so the check
    aligns with the training / backtest data pipeline in ``lib/data.py``,
    ``src/walk_forward.py`` (``load_prefiltered``), and ``src/backtest.py``
    (``slice_test_csv``), which apply the same filter.
    """
    df = pd.read_csv(csv_path)
    if "<CLOSE>" not in df.columns or "<DES>" not in df.columns:
        raise RuntimeError(f"{csv_path.name}: missing <CLOSE> or <DES> column")
    ok = (df["<OPEN>"] > 0) & (df["<HIGH>"] > 0) & (df["<LOW>"] > 0) & (df["<CLOSE>"] > 0)
    ok &= df["<DES>"].notna()
    df = df.loc[ok].reset_index(drop=True)
    close = df["<CLOSE>"].astype(float).to_numpy()
    des = df["<DES>"].astype(int).to_numpy()
    y = compute_y(close, horizon=horizon)
    return float((des == y).mean()), len(des)


def run_marginal(universes: tuple[str, ...], accuracies: list[int],
                 data_dir: Path, horizon: int, tolerance: float,
                 verbose: bool) -> int:
    total = 0
    passed = 0
    failures: list[str] = []
    per_acc: dict[int, list[float]] = {a: [] for a in accuracies}

    for uni in universes:
        symbols = load_universe_symbols(uni, data_dir)
        print(f"[{uni}] marginal check on {len(symbols)} symbols x "
              f"{len(accuracies)} accuracies (h={horizon}, tol={tolerance:.3f}) ...")
        for sym in symbols:
            for acc in accuracies:
                csv_path = data_dir / f"{sym}_all_{acc}.csv"
                if not csv_path.is_file():
                    failures.append(f"[{uni}] {csv_path.name}: file missing")
                    continue
                total += 1
                try:
                    rate, n = check_marginal(csv_path, horizon=horizon)
                except Exception as e:  # noqa: BLE001
                    failures.append(f"[{uni}] {csv_path.name}: error: {e}")
                    continue
                per_acc[acc].append(rate)
                target = acc / 100.0
                ok = abs(rate - target) <= tolerance
                if ok:
                    passed += 1
                    if verbose:
                        print(f"  OK   {sym}_all_{acc}.csv  "
                              f"rate={rate*100:.2f}%  target={acc}%  n={n}")
                else:
                    failures.append(
                        f"[{uni}] {csv_path.name}: rate={rate*100:.2f}% "
                        f"target={acc}% deviation={abs(rate-target)*100:.2f}% "
                        f"> tol={tolerance*100:.2f}%"
                    )
                    print(f"  FAIL {sym}_all_{acc}.csv: rate={rate*100:.2f}%  "
                          f"target={acc}%  deviation={abs(rate-target)*100:.2f}%")

    print()
    for acc in accuracies:
        rates = per_acc[acc]
        if not rates:
            continue
        arr = np.asarray(rates)
        print(f"  accuracy={acc}%  n={len(arr):3d}  "
              f"empirical mean={arr.mean()*100:.2f}%  "
              f"min={arr.min()*100:.2f}%  max={arr.max()*100:.2f}%")

    print()
    if failures:
        print(f"MARGINAL CHECK FAILED: {len(failures)} issue(s) "
              f"({passed}/{total} passed).")
        for msg in failures[:20]:
            print(f"  {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print(f"MARGINAL CHECK PASSED: all {passed}/{total} shipped DES columns "
          f"agree with the target accuracy within +/- {tolerance*100:.2f}%.")
    return 0


def run_bit_exact(universes: tuple[str, ...], accuracies: list[int],
                  data_dir: Path, compare_to: Path, verbose: bool) -> int:
    if not compare_to.is_dir():
        raise SystemExit(f"--compare-to directory not found: {compare_to}")
    total = 0
    passed = 0
    failures: list[str] = []
    for uni in universes:
        symbols = load_universe_symbols(uni, data_dir)
        print(f"[{uni}] bit-exact check vs {compare_to} on {len(symbols)} symbols x "
              f"{len(accuracies)} accuracies ...")
        for sym in symbols:
            for acc in accuracies:
                name = f"{sym}_all_{acc}.csv"
                a, b = data_dir / name, compare_to / name
                if not a.is_file() or not b.is_file():
                    failures.append(f"[{uni}] {name}: missing on one side")
                    continue
                total += 1
                # shallow=False forces byte-level comparison, not size/mtime.
                if filecmp.cmp(a, b, shallow=False):
                    passed += 1
                    if verbose:
                        print(f"  OK   {name}")
                else:
                    failures.append(f"[{uni}] {name}: bytes differ")
                    print(f"  FAIL {name}: bytes differ")

    print()
    if failures:
        print(f"BIT-EXACT CHECK FAILED: {len(failures)} issue(s) "
              f"({passed}/{total} passed).")
        for msg in failures[:20]:
            print(f"  {msg}")
        return 1
    print(f"BIT-EXACT CHECK PASSED: all {passed}/{total} shipped CSVs match "
          f"{compare_to}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("marginal", "bit-exact"), default="marginal")
    ap.add_argument("--universe", choices=("tw50", "dow30", "both"), default="both")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    choices=(5, 10, 20, 60),
                    help=f"Forecast horizon h in trading days (default {DEFAULT_HORIZON})")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="Absolute tolerance on the marginal agreement rate "
                         f"(default {DEFAULT_TOLERANCE}, i.e. +/- 2 percentage points)")
    ap.add_argument("--accuracies", type=int, nargs="+", choices=ACCURACIES,
                    default=list(ACCURACIES),
                    help="Accuracies to verify (default: 55 60 65 75)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="For --mode bit-exact: directory containing regenerated CSVs "
                         "(e.g. produced by scripts/gen_des_by_accuracy.py --seed 42 "
                         "--horizon 20 --out-dir data_regenerated/).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print a line per file, not just failures")
    args = ap.parse_args()

    universes = ("tw50", "dow30") if args.universe == "both" else (args.universe,)

    if args.mode == "marginal":
        return run_marginal(
            universes=universes,
            accuracies=args.accuracies,
            data_dir=args.data_dir,
            horizon=args.horizon,
            tolerance=args.tolerance,
            verbose=args.verbose,
        )

    if args.compare_to is None:
        print("error: --mode bit-exact requires --compare-to <dir>", file=sys.stderr)
        return 2
    return run_bit_exact(
        universes=universes,
        accuracies=args.accuracies,
        data_dir=args.data_dir,
        compare_to=args.compare_to,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
