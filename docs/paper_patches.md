# Paper text patches (ready for LaTeX paste)

This document consolidates the small caption / paragraph edits needed in the
ACM ICAIF 2026 submission to keep the paper text consistent with the shipped
repository. Nothing here changes an experimental number; every patch either
narrows a claim, names an exclusion, or points the reader at a reproducibility
artifact that now exists.

The line references below assume the current LaTeX source; if the numbering
has drifted, search for the highlighted phrase.

---

## Patch 1 — Table 4 caption (2150 excluded)

**Current** (paraphrased):
> Table 4. Per-symbol excess return over buy-and-hold on the 50 TW50
> constituents...

**Replace with**:

```latex
\caption{Per-symbol excess return over buy-and-hold on the TW50 universe
(N = 50 constituents as of 2023-12-29, symbol 2150 excluded — see
Appendix / \texttt{docs/coverage\_report.md} in the released code).
Test window 2024-01-02 to 2026-03-30, TW retail costs 0.10\,\%/0.34\,\%.
The four accuracy tiers $\rho \in \{55, 60, 65, 75\}\,\%$ correspond to the
target directional accuracy of the DES signal, not to a look-back window.}
```

**Rationale.** The paper narrative currently cites `2150` (崇越電) inline as
a TW50 constituent. The 2023-12-29 constituent list actually contains 50
tickers with 2150 already dropped, so no checkpoint was ever produced for it.
Naming the exclusion in the caption avoids reviewer confusion when they cross
reference `backtest_summary.csv`.

---

## Patch 2 — Figure 5(c) caption (75 % accuracy uses 14 symbols)

**Current** (paraphrased):
> Figure 5(c). Distribution of DQN excess return at $\rho = 75\,\%$ ...

**Replace with**:

```latex
\caption{Distribution of DQN excess return at $\rho = 75\,\%$ target
directional accuracy. Marker cloud spans the \textbf{14 of 50 TW50 symbols}
that have a shipped 75\,\% checkpoint (1101, 2382, 2603, 2801, 2880, 2883,
2884, 2885, 2886, 2887, 2890, 3045, 5880, 6505); the smaller sample size
relative to Figure 5(a,b) reflects checkpoint coverage, not a filtering step
on returns. Dow30 (Figure 5(d)) has full 30-of-30 coverage at $\rho = 75\,\%$.}
```

**Rationale.** Reviewer 2's concern was that panel (c) looks visually sparser
than (a,b); documenting the 14-symbol basis explains the visual asymmetry.

---

## Patch 3 — Figure 7 caption (frontier basis)

**Current** (paraphrased):
> Figure 7. Efficient-frontier view of DQN portfolios across accuracy tiers.

**Replace with**:

```latex
\caption{Efficient-frontier view of DQN portfolios across accuracy tiers
$\rho \in \{55, 60, 65, 75\}\,\%$ on the 2024-01-02 to 2026-03-30 test
window. Each marker aggregates a per-symbol backtest into an equal-weighted
(or cap-weighted, marker shape) portfolio. Marker counts per tier: TW50
25 / 25 / 25 / \textbf{14} symbols and Dow30 0 / 30 / 30 / 30 symbols at
$\rho = 55 / 60 / 65 / 75\,\%$ respectively; the 75\,\% frontier is
therefore estimated from a smaller cohort on the TW50 side. See
\texttt{docs/coverage\_report.md} for the full per-symbol coverage matrix.}
```

**Rationale.** The frontier "opens up" at $\rho = 75\,\%$ partly because the
sample shrinks to the higher-quality subset that received the extra training
budget. Naming this in the caption pre-empts a survivorship-bias concern.

---

## Patch 4 — Methods §DES construction (seed disclosure)

**Insert one sentence** at the end of the paragraph that introduces the DES
noise model (Eq. 2):

```latex
The shipped CSVs (\texttt{data/<sym>\_all\_<\rho>.csv}) were generated with
seed 42 (NumPy \texttt{default\_rng(42)}) at horizon $h = 20$ trading days;
downstream users can bit-exactly reproduce them via
\texttt{scripts/gen\_des\_by\_accuracy.py --seed 42} and confirm marginal
agreement rates with \texttt{scripts/verify\_des.py} (all 320 shipped columns
pass at a 9\,pp tolerance; see the released code).
```

**Rationale.** Issue #5 in the internal review — the DES seed was not stated
anywhere in the paper text, making the label channel effectively unverifiable
from the manuscript alone.

---

## Patch 5 — Methods §FinRL Figure 8 (reproducibility line)

**Insert one sentence** at the end of the paragraph describing the 5 SB3
variants:

```latex
The pre-trained SB3 checkpoints are not distributed (~2\,GB, external
package-version drift); the driver script
\texttt{finrl/backtest/run\_all.sh} (and its PowerShell twin
\texttt{run\_all.ps1}) rebuilds every variant end-to-end (prep\,\textrightarrow\,train\,\textrightarrow\,backtest) under a single
\texttt{FINRL\_SEED} env var (default 42).
```

**Rationale.** Issue #6 in the internal review — Figure 8 was flagged as
"not reproducible from the release" because the SB3 zips are absent. Naming
the driver script closes that gap.

---

## Cross-reference to the repo

- Table 4 exclusion → [docs/coverage_report.md](coverage_report.md#known-omissions-documented-not-bugs)
- Figure 5(c) / 7 basis → [docs/coverage_report.md](coverage_report.md#figure-7-frontier-at-rho--75--basis)
- Patch 4 seed & verifier → `scripts/verify_des.py`
- Patch 5 driver → `finrl/backtest/run_all.sh`, `finrl/backtest/run_all.ps1`
