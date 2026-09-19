# Phase 1 — research suite

Reproduces every Phase 1 result. Deterministic: all random procedures (stationary
bootstrap, sector-label permutation test) use fixed seeds, and repeated runs
produce byte-identical artifacts.

```bash
python -m pip install -r requirements.txt
python research/run_phase1.py --all
```

Runtime ≈ 6 minutes. Console output is teed to `artifacts/phase1/phase1_log.txt`.

## Modules

| File | Purpose |
|---|---|
| `src/data.py` | Panel reshaping, forward-return decomposition, neutralisation, and a mirror of the engine's cost model |
| `src/icstats.py` | Newey–West HAC t-statistics, stationary bootstrap, Benjamini–Hochberg FDR, Lo–MacKinlay variance ratio, IC and tercile machinery |
| `research/phase1_00_integrity.py` | Invariant checks and look-ahead audit: every derived field recomputed from raw data using only observations ≤ t |
| `research/phase1_01_structure.py` | Factor count vs the Marchenko–Pastur edge, sector-block permutation test, variance-ratio tests, return-segment decomposition |
| `research/phase1_02_signal_screen.py` | All 19 candidate fields × 3 neutralisations, FDR-controlled, with segment decomposition and tercile spreads |
| `research/phase1_03_diagnostics.py` | Decay profile, sub-period stability, conditional strength, orthogonality, and the cost bridge |
| `research/build_notebook.py` | Generates `analysis.ipynb` from source; `--run` executes it in place |

## The notebook

`analysis.ipynb` is the research narrative: the same analysis as a linear, annotated
read-through with six charts and four adversarial tests the scripts do not run
(signal-lag placebo, within-date return shuffle, leave-one-asset-out,
leave-one-year-out). It imports helpers from `src/` rather than restating them, so it
cannot drift from the code the strategy uses, and it calibrates each estimator against
synthetic data with a known answer before trusting it.

Regenerate and re-execute with:

```bash
python research/build_notebook.py --run
```

The notebook is not executed by the evaluator.

## Methodological commitments

**Signals are scored against `open(t+1) → close(t+1)`.** The engine grants a newly
requested position only the intraday move; the overnight gap accrues solely to
positions already held. Scoring against close-to-close returns overstates what is
capturable and misattributes where the edge sits.

**t-statistics are Newey–West corrected.** Daily ICs are serially correlated, so
naive standard errors overstate significance. Both are reported side by side.

**The screen is FDR-controlled.** 56 live tests were run. Benjamini–Hochberg at
q = 0.10 gives a cutoff of p ≤ 0.0125; promoting a field on its raw p-value would
not be defensible at this test count.

**Fields with no cross-sectional variation after neutralisation are excluded
explicitly** rather than silently returning NaN (`Panel.is_degenerate`).

**Costs use the engine's own arithmetic.** `src/data.one_way_cost_rate` mirrors
`starter.backtester.trading_cost`, including the detail that costs are priced off
the *decision* date's close row — the engine passes `data.iloc[(i-1)*n : i*n]`,
not the execution date's row.
