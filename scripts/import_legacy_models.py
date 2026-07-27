"""Import legacy trained DQN checkpoints into the public repo.

Legacy layout (private):
    d:/DRL/saves/{sym}_{window}_val20240101_20260331_okohlcv_*_novol*/best_val-<reward>.data

For each TW50 symbol x window in {55,60,65,75}, pick the checkpoint with the
highest ``val_reward`` (parsed from the filename ``best_val-<reward>.data``),
copy it to ``trained_models/{sym}_all_{window}.data``, and write a manifest.

Note: these legacy checkpoints were trained with a fixed 2005~2023 train /
2024-01~2026-03 validation split, NOT the 5-fold walk-forward CV used by
``src/train_dqn.py`` in this public repo. See README for details.

Usage:
    python scripts/import_legacy_models.py
    python scripts/import_legacy_models.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_ROOT = Path(r"d:/DRL/saves")
WINDOWS = (55, 60, 65, 75)
CKPT_RE = re.compile(r"^best_val-(-?\d+(?:\.\d+)?)\.data$")
FOLDER_RE = re.compile(
    r"^(?P<sym>\d+)_(?P<win>\d+)_val20240101_20260331_okohlcv_.*_novol(?:_.*)?$"
)


def load_tw50_symbols(constituents_csv: Path) -> list[str]:
    symbols: list[str] = []
    with constituents_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            symbols.append(row[1].strip())
    if len(symbols) != 50:
        raise RuntimeError(f"Expected 50 TW50 constituents, got {len(symbols)}")
    return symbols


def scan_legacy(legacy_root: Path) -> dict[tuple[str, int], list[tuple[float, Path, Path]]]:
    """Return {(sym, window): [(val_reward, ckpt_path, folder), ...]}."""
    found: dict[tuple[str, int], list[tuple[float, Path, Path]]] = {}
    if not legacy_root.is_dir():
        raise SystemExit(f"legacy_root not found: {legacy_root}")
    for folder in legacy_root.iterdir():
        if not folder.is_dir():
            continue
        m = FOLDER_RE.match(folder.name)
        if not m:
            continue
        sym = m.group("sym")
        window = int(m.group("win"))
        if window not in WINDOWS:
            continue
        for ckpt in folder.glob("best_val-*.data"):
            cm = CKPT_RE.match(ckpt.name)
            if not cm:
                continue
            reward = float(cm.group(1))
            found.setdefault((sym, window), []).append((reward, ckpt, folder))
    return found


def pick_best(candidates: list[tuple[float, Path, Path]]) -> tuple[float, Path, Path]:
    return max(candidates, key=lambda t: t[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    ap.add_argument("--constituents", type=Path,
                    default=REPO_ROOT / "data" / "tw50_2023-12-29.csv")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "trained_models")
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "trained_models" / "manifest.csv")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    symbols = load_tw50_symbols(args.constituents)
    print(f"TW50: {len(symbols)} symbols x {len(WINDOWS)} windows = {len(symbols) * len(WINDOWS)} targets")
    found = scan_legacy(args.legacy_root)
    print(f"Scanned legacy root: {len(found)} (sym,window) pairs found")

    args.dest.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    n_copied = n_missing = n_skipped = 0
    for sym in symbols:
        for w in WINDOWS:
            key = (sym, w)
            dst = args.dest / f"{sym}_all_{w}.data"
            row = {"symbol": sym, "window": w, "val_reward": "",
                   "source_folder": "", "source_ckpt": "",
                   "dest": dst.relative_to(REPO_ROOT).as_posix(), "notes": ""}
            if key not in found:
                row["notes"] = "MISSING legacy checkpoint"
                manifest_rows.append(row)
                n_missing += 1
                continue
            reward, ckpt, folder = pick_best(found[key])
            row["val_reward"] = f"{reward:.4f}"
            row["source_folder"] = folder.name
            row["source_ckpt"] = ckpt.name
            if dst.exists() and not args.overwrite:
                row["notes"] = "exists (kept)"
                manifest_rows.append(row)
                n_skipped += 1
                continue
            if args.dry_run:
                row["notes"] = "dry-run"
                manifest_rows.append(row)
                continue
            shutil.copy2(ckpt, dst)
            manifest_rows.append(row)
            n_copied += 1

    if not args.dry_run:
        with args.manifest.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()),
                                lineterminator="\n")
            wr.writeheader()
            wr.writerows(manifest_rows)

    print(f"\nCopied     : {n_copied}")
    print(f"Kept       : {n_skipped}")
    print(f"Missing    : {n_missing}")
    print(f"Manifest   : {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
