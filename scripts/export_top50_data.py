"""Export TW50 (2023-12-29 constituents) OHLCV+DES CSVs to the public repo,
dropping the ``<VOLUME>`` column so only ``<DES> + <OHLC>`` remains.

Source : d:/DRL/data/{sym}_all_{window}.csv    (has <DATE> <DES> <OHLCV>)
Target : public_tw_dqn/data/{sym}_all_{window}.csv  (drops <VOLUME>)

Usage:
    python scripts/export_top50_data.py
    python scripts/export_top50_data.py --source d:/DRL/data --overwrite
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"d:/DRL/data")
WINDOWS = (55, 60, 65, 75)
KEEP_COLS = ("<DATE>", "<DES>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>")


def load_tw50_symbols(constituents_csv: Path) -> list[str]:
    symbols: list[str] = []
    with constituents_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            sym = row[1].strip()
            if sym:
                symbols.append(sym)
    if len(symbols) != 50:
        raise RuntimeError(
            f"Expected 50 TW50 constituents, got {len(symbols)} from {constituents_csv}"
        )
    return symbols


def _iter_rows(csv_path: Path):
    """Yield rows re-projected onto KEEP_COLS from a single source CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        missing = [c for c in KEEP_COLS if c not in header]
        if missing:
            raise RuntimeError(f"{csv_path.name}: missing columns {missing}")
        indices = [header.index(c) for c in KEEP_COLS]
        for row in reader:
            yield [row[i] for i in indices]


def write_projection(sources: list[Path], dst: Path) -> tuple[int, int]:
    """Concatenate one or more sources into ``dst`` keeping KEEP_COLS."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dst.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, lineterminator="\n")
        writer.writerow(list(KEEP_COLS))
        for src in sources:
            for row in _iter_rows(src):
                writer.writerow(row)
                n += 1
    return n, len(KEEP_COLS)


def resolve_sources(source_dir: Path, sym: str, window: int) -> list[Path]:
    """Prefer ``{sym}_all_{w}.csv`` when it has DES; otherwise reconstruct from
    ``train_before_val_..._{w}.csv`` + ``val_..._{w}.csv``."""
    main = source_dir / f"{sym}_all_{window}.csv"
    if main.is_file():
        with main.open("r", encoding="utf-8", newline="") as f:
            hdr = next(csv.reader(f))
        if "<DES>" in hdr:
            return [main]
    train = source_dir / f"{sym}_train_before_val_20240101_20260331_okohlcv_{window}.csv"
    val = source_dir / f"{sym}_val_20240101_20260331_okohlcv_{window}.csv"
    if train.is_file() and val.is_file():
        return [train, val]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="Directory containing {sym}_all_{window}.csv originals")
    ap.add_argument("--constituents", type=Path,
                    default=None,
                    help="TW50 constituent CSV (default: <source>/tw50_2023-12-29.csv)")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "data",
                    help="Destination directory (public repo data/)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rewrite files even if they already exist")
    args = ap.parse_args()

    constituents = args.constituents or (args.source / "tw50_2023-12-29.csv")
    symbols = load_tw50_symbols(constituents)
    print(f"TW50 constituents ({len(symbols)}): {symbols}")

    args.dest.mkdir(parents=True, exist_ok=True)

    # Copy the constituent list too.
    dest_list = args.dest / constituents.name
    if args.overwrite or not dest_list.exists():
        dest_list.write_bytes(constituents.read_bytes())
        print(f"Copied {constituents.name} -> {dest_list.relative_to(REPO_ROOT)}")

    missing_src: list[str] = []
    n_written = 0
    for sym in symbols:
        for w in WINDOWS:
            dst = args.dest / f"{sym}_all_{w}.csv"
            if dst.exists() and not args.overwrite:
                continue
            sources = resolve_sources(args.source, sym, w)
            if not sources:
                missing_src.append(f"{sym}_all_{w} (no usable source)")
                continue
            rows, cols = write_projection(sources, dst)
            n_written += 1
            src_desc = sources[0].name if len(sources) == 1 else f"{sources[0].name} + {sources[1].name}"
            print(f"  {sym}_all_{w}: {rows} rows, {cols} cols  <- {src_desc}")

    print(f"\nWrote {n_written} CSVs to {args.dest}")
    if missing_src:
        print(f"\nWARNING: {len(missing_src)} source files missing:")
        for p in missing_src:
            print(f"  {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
