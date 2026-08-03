# Reviewer reproducibility kit

This folder is the falsifiable half of the paper's anti-fabrication argument
for [tunglich/Market-Timing-DQN](https://github.com/tunglich/Market-Timing-DQN):
the six shipped result CSVs listed in `MANIFEST.sha256` are the exact bytes
that produced every headline number reported in *"How Accurate Must Equity
Market Timing Signals Be?"*. If anyone edits them after publication, the
[CI badge](https://github.com/tunglich/Market-Timing-DQN/actions/workflows/ci.yml)
turns red on the next push.

## What is fingerprinted

`MANIFEST.sha256` is a plain `sha256sum` file with one row per shipped CSV
(SHA-256 of the LF-normalized git blob — matches `git show :<file> | sha256sum`
on any platform):

| file | role |
| --- | --- |
| `backtest_summary.csv`           | TW50: per-stock/per-accuracy best-fold backtest summary. |
| `backtest_summary_dow30.csv`     | Dow30 replication of the above. |
| `portfolio_summary.csv`          | TW50: portfolio-level headline metrics (return, Sharpe, MDD, WR). |
| `portfolio_summary_dow30.csv`    | Dow30 replication. |
| `portfolio_timeseries.csv`       | TW50: daily equity curve of the model portfolio vs benchmark. |
| `portfolio_timeseries_dow30.csv` | Dow30 replication. |

## 60-second reviewer verification

```bash
# Linux / macOS / WSL
sha256sum -c reproducibility/MANIFEST.sha256
# expected: "6/6 files: OK" (any tamper => "FAILED" on the offending row)
```

```powershell
# Windows PowerShell (uses git's LF blob so it matches the manifest even
# under core.autocrlf=true)
git ls-files 'backtest_summary*.csv' 'portfolio_summary*.csv' 'portfolio_timeseries*.csv' |
  ForEach-Object {
    $lf   = & git show ":$_"
    $hash = (Get-FileHash -Algorithm SHA256 -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes(($lf -join "`n") + "`n")))).Hash.ToLower()
    "$hash  $_"
  } |
  Compare-Object -ReferenceObject (Get-Content reproducibility/MANIFEST.sha256) -DifferenceObject { $_ } -PassThru |
  ForEach-Object { "MISMATCH: $_" }
```

## Regenerating the manifest

Only re-run this after you deliberately re-run the pipeline and re-check the
paper numbers:

```bash
bash scripts/regen_manifest.sh
```

The GitHub Actions `fast-checks` job (see `.github/workflows/ci.yml`) runs
`sha256sum -c` on every push to `main` and on every pull request.
