"""Closed-form directional-accuracy -> F1 / AUC mapping (Eq. 6 of the paper).

Under the symmetric independent-flip label model of the paper (Eq. 2), the
hypothetical predictor emits its ground-truth label with probability rho and
the opposite label with probability 1 - rho. This yields, irrespective of the
up-label base rate ``pi = Pr(y = +1)``,

    TPR = rho,   FPR = 1 - rho,
    AUC = 0.5 * (TPR + TNR) = rho,
    F1  = 2 * pi * rho / (2 * pi * rho + 1 - rho).

F1 depends on the base rate ``pi``; AUC does not. This CLI implements the
mapping and prints the F1 / AUC values that machine-learning practitioners
can screen a candidate binary classifier against before running a full
market-timing backtest.

Usage:
    python scripts/accuracy_to_metrics.py --accuracy 0.65 --base-rate 0.53
    python scripts/accuracy_to_metrics.py --grid --base-rate 0.53
    python scripts/accuracy_to_metrics.py --grid --out reports/accuracy_map.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def accuracy_to_metrics(rho: float, base_rate: float) -> dict[str, float]:
    if not (0.5 <= rho <= 1.0):
        raise ValueError(f"accuracy must be in [0.5, 1.0], got {rho}")
    if not (0.0 < base_rate < 1.0):
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    tpr = rho
    fpr = 1.0 - rho
    tnr = 1.0 - fpr
    auc = 0.5 * (tpr + tnr)
    f1 = 2.0 * base_rate * rho / (2.0 * base_rate * rho + (1.0 - rho))
    return {"accuracy": rho, "base_rate": base_rate, "tpr": tpr, "fpr": fpr,
            "tnr": tnr, "auc": auc, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accuracy", type=float, default=None,
                    help="Single directional accuracy in [0.5, 1.0]")
    ap.add_argument("--base-rate", type=float, default=0.53,
                    help="Up-label base rate pi (default 0.53 = TWSE Top-50 test window)")
    ap.add_argument("--grid", action="store_true",
                    help="Print the mapping over 0.50..0.90 in 0.01 steps")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional CSV output path when --grid is set")
    args = ap.parse_args()

    if args.grid:
        rows = [accuracy_to_metrics(round(0.50 + k * 0.01, 2), args.base_rate)
                for k in range(41)]
        print(f"{'rho':>6}  {'base':>5}  {'AUC':>6}  {'F1':>6}")
        for r in rows:
            print(f"{r['accuracy']:>6.2f}  {r['base_rate']:>5.2f}  "
                  f"{r['auc']:>6.3f}  {r['f1']:>6.3f}")
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            print(f"\nWrote {args.out}  ({len(rows)} rows)")
        return 0

    if args.accuracy is None:
        ap.error("either --accuracy or --grid is required")
    r = accuracy_to_metrics(args.accuracy, args.base_rate)
    print(f"accuracy rho = {r['accuracy']:.3f}")
    print(f"base rate pi = {r['base_rate']:.3f}")
    print(f"TPR = {r['tpr']:.3f}   FPR = {r['fpr']:.3f}   TNR = {r['tnr']:.3f}")
    print(f"AUC = {r['auc']:.3f}")
    print(f"F1  = {r['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
