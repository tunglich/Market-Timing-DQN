#!/usr/bin/env bash
# Regenerate reproducibility/MANIFEST.sha256 from LF-normalized git blobs.
# The six root CSVs are the falsifiable paper results a reviewer would want
# to verify have not been fabricated after publication.
set -e
cd "$(dirname "$0")/.."
mkdir -p reproducibility
{
  git ls-files 'backtest_summary*.csv' 'portfolio_summary*.csv' 'portfolio_timeseries*.csv' \
    | sort \
    | while IFS= read -r f; do
        h=$(git show ":$f" | sha256sum | cut -d' ' -f1)
        printf '%s  %s\n' "$h" "$f"
      done
} > reproducibility/MANIFEST.sha256
wc -l reproducibility/MANIFEST.sha256
