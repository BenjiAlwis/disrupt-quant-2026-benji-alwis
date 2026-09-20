# Disrupt Quant 2026 — Benji Alwis

**Candidate:** Benji Alwis
**Strategy:** Smoothed sector-neutral carry

Each close, the strategy averages `carry_score` over the last 21 sessions. It ranks the four assets inside each sector, centres the ranks and scales the book to 120% gross, so the book is dollar- and sector-neutral at rebalance. A sector is traded only when a held weight has drifted more than a quarter of a rank step from its target. The research found a stronger signal, one-day reversal, but it was rejected: it turns over about 40× as much, and costs eat all of its edge.

The two-page argument is in [`research_note.pdf`](research_note.pdf). This README covers how to reproduce every number in it, and how the work was done.

---

## 1. Reproduce

### Environment

The evaluation environment is Python 3.13 with the pinned packages in `requirements.txt`: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2 and matplotlib 3.10.7. The strategy itself imports only numpy and pandas. Results are deterministic, and runs under Python 3.13.7 and 3.11 produced byte-identical `daily.csv` files.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### The submission (official engine, about 5 s per run)

```bash
python starter/backtester.py                                        # development, 1x costs
python starter/backtester.py --cost-multiplier 1.5
python starter/backtester.py --cost-multiplier 2
python starter/backtester.py --split validation --out results/validation_x1.0
python starter/backtester.py --split validation --cost-multiplier 1.5 --out results/validation_x1.5
python starter/backtester.py --split validation --cost-multiplier 2 --out results/validation_x2.0
python validate_submission.py
```

The committed files in `artifacts/validation/` are copies of those validation outputs, renamed `*_x{1.0,1.5,2.0}`. Re-running the commands above reproduces them byte for byte.

### The research, in the order it was done

| Step | Command | Writes |
|---|---|---|
| Phase 1: data integrity, structure, signal screen, diagnostics | `python research/run_phase1.py --all` | `artifacts/phase1/` |
| Phase 2.0: reversal economics on 2021–22 only | `python research/phase2/p2_00_calibration_window.py` | console |
| Phase 2.1: design check on 2021-07 → 2022-12 | `python research/phase2/p2_01_design_check.py` | `trial_log.csv` |
| Phase 2.2: pre-registered walk-forward, 2023H1 / 2023H2 / 2024H1 | `python research/phase2/p2_02_walk_forward.py` | `walkforward_*.csv` |
| Phase 2.3: does the overlay add anything out of sample? | `python research/phase2/p2_03_overlay_increment.py` | console |
| Phase 2.4: one-at-a-time sensitivity (reported, never used to select) | `python research/phase2/p2_04_sensitivity.py` | `sensitivity.csv` |
| Phase 2.5: robustness, attribution, bootstrap, deflated Sharpe | `python research/phase2/p2_05_robustness.py` | `carry_blocks.csv`, `carry_fulldev_*.csv` |
| Phase 2.6: choose the carry de-noising rule on 2021-07 → 2022-12 only | `python research/phase2/p2_07_carry_denoise_design.py` | `sensitivity_carry_smooth.csv` |
| Phase 2.7: every remaining development number in the note | `python research/phase2/p2_08_note_numbers.py` | `carry_denoise_confirmation.csv`, `note_tradability_table.json`, `sensitivity_gross_final.json` |
| Validation diagnostics (descriptive; runs no strategy) | `python research/validation_diagnostics.py` | console |

Output paths are under `artifacts/phase2/` unless shown otherwise. Phase 1 takes about 6 minutes; each Phase 2 script takes a few minutes or less. All random procedures use fixed seeds.

`p2_01`–`p2_04` and `p2_07` append one row per backtest to `artifacts/phase2/trial_log.csv`, which is the honest trial count behind the deflated Sharpe. Re-running them adds rows. Run `git checkout artifacts/phase2/trial_log.csv` to restore the committed log. `p2_05`, `p2_06` and `p2_08` rerun configurations that were already logged, so they do not append.

`harness.py` asserts that it never loads validation data. Only the three validation commands above and `research/validation_diagnostics.py` read `data/validation.csv`.

### Notebook and note (not needed by the evaluator)

`analysis.ipynb` is the Phase 1 research narrative. It has calibrated estimators, six charts and four adversarial tests: signal-lag placebo, return shuffle, leave-one-asset-out and leave-one-year-out. It needs `pip install nbformat nbclient ipykernel`.

`research/build_notebook.py --run` regenerates the original machine-built version as `research/analysis_generated.ipynb`. The root notebook was then edited by hand, so the generator never overwrites it.

`research_note.pdf` is the submitted note. `research/build_research_note.py` (reportlab, Carlito font) generates its figure and the draft layout, and its tables are read from artifacts. The final wording was edited by hand, so rebuilding it reproduces the draft text, not the submitted text.

---

## 2. Results

All figures come from the official engine (`starter/backtester.py`). Each period starts from $1m cash and liquidates at its last close. Returns are annualised, and `transaction_costs` is the engine's sum of daily cost fractions over the period.

| Period | Dates | Sessions | Ann. return | Vol | Sharpe 1× / 1.5× / 2× | Max DD | Turnover/yr | Costs (total) |
|---|---|---|---|---|---|---|---|---|
| Development | 2021-01-04 → 2024-06-28 | 910 | 8.84% | 5.18% | 1.66 / 1.61 / 1.57 | −3.39% | 8.3 | 1.77% |
| Walk-forward (continuous) | 2023-01-01 → 2024-06-28 | 390 | 12.57% | — | 2.44 / 2.39 / 2.35 | −2.45% | 8.3 | — |
| **Public validation (single run)** | **2024-07-01 → 2024-12-31** | **132** | **−1.33%** (−0.70% total) | 5.41% | **−0.22 / −0.28 / −0.35** | **−5.17%** | 11.1 | 0.36% |
| Reference: equal-weight long-only | 2021-01-04 → 2024-06-28 | 910 | 4.2% | — | 0.42 (1×) | −11.0% | 2.4 | — |

Every half-year block of the submitted configuration is shown below. The design window is where the rules were chosen. The walk-forward blocks are out of sample for the construction but not for the discovery of `carry_score` (see §4).

| Block | Role | Ann. return | Sharpe 1× / 1.5× / 2× | Max DD |
|---|---|---|---|---|
| 2021-07 → 2021-12 | design | 2.4% | 0.57 / 0.49 / 0.40 | −1.9% |
| 2022-01 → 2022-06 | design | 1.5% | 0.31 / 0.25 / 0.19 | −2.9% |
| 2022-07 → 2022-12 | design | 10.5% | 2.22 / 2.13 / 2.05 | −1.7% |
| 2023-01 → 2023-06 | walk-forward | 6.6% | 1.28 / 1.21 / 1.15 | −2.5% |
| 2023-07 → 2023-12 | walk-forward | 18.2% | 3.33 / 3.27 / 3.22 | −1.4% |
| 2024-01 → 2024-06 | walk-forward | 11.7% | 2.49 / 2.43 / 2.36 | −2.0% |

Sources:
- `results/development/metrics.json` (gitignored; written by the commands in §1)
- `artifacts/validation/metrics_x*.json`
- `artifacts/phase2/carry_denoise_confirmation.csv`
- `artifacts/phase2/carry_blocks.csv`

**Behaviour in validation.** The engine made zero constraint adjustments. Maximum |net| exposure was 2.9%, beta was −0.02 and average gross exposure was 119%. The traded carry IC fell to +0.013 (t = 0.63) from +0.028 in development: weaker, but not reversed. The Sharpe of −0.22 falls in the bottom 2.2% of overlapping 132-session development windows. The −5.17% drawdown is deeper than any development window of that length, where the worst was −3.39%. That was one of the three warning signs stated in advance. The full assessment is in `artifacts/validation/run_record.json`.

**Sensitivity.**
- **Costs:** Sharpe falls from 1.66 to 1.57 when costs double.
- **Gross exposure:** development Sharpe is 1.66 / 1.66 / 1.67 at 0.8 / 1.2 / 1.6 gross, so gross is a risk budget, not a tuned parameter.
- **Everything else:** the one-parameter-at-a-time grid is in `artifacts/phase2/sensitivity.csv`.

---

## 3. How the strategy was built

**Phase 1: what is real?**
- **Integrity.** All 15 derived fields were rebuilt from raw prices using only data up to date t. The check found no look-ahead.
- **Screen.** Signals were scored against the return a new position can actually earn, which is open(t+1) → close(t+1). The screen covered 37 raw and sector-neutral tests over 19 fields, with Newey–West t-statistics and FDR/Bonferroni control. Only two signals survived Bonferroni: one-day reversal (IC +0.054, t 5.84) and `carry_score` (IC +0.030, t 4.32).
- **Diagnostics.** These covered decay, sub-period stability, placebo tests and a cost bridge.

**Phase 2: can either signal be traded?** The configuration and its expected results were written into `artifacts/phase2/preregistration.json` before the 2023–24 blocks were run. Two documented deviations followed.

1. **The reversal overlay was dropped** (`decision_record.json`). It failed its pre-registered test: it added return in 1 of 3 blocks (full-period increment +1.0%/yr, t 0.64), and the increment was negative at 1.5× and 2× costs.
2. **Carry smoothing was added** (`decision_record_2_carry_denoise.json`). A comparison with an external AI-generated candidate prompted the idea that daily carry ranks were noisy. The rule itself was chosen on 2021-07 → 2022-12 only, using a criterion declared before running: highest Sharpe at 1.5× costs, with a preference for smoothing and for the shorter window. 58% of daily rank changes reversed within five sessions. Smoothing cut the share of sessions with a rank change from about 70% to 24%, and turnover from 31×/yr to 8×/yr.

**Freeze and validate.** The strategy was frozen and committed (`d03918c`), then run once on validation at 1×, 1.5× and 2× costs (`450fb72`). Nothing changed afterwards. The rejected alternatives (daily ranks, the reversal overlay, the external candidate) were deliberately **not** run on validation, so it could never become a model-selection set. The validation plan is in `validation_plan.json`; v1 is kept as `validation_plan__superseded_v1.json`.

**Nothing was edited after validation, including comments.** The docstrings at the top of `strategy.py` and `src/strategy_core.py` were written at pre-registration. They still describe the "carry core plus cost-gated reversal overlay" design, and the `Config` defaults still have the overlay on. The submitted behaviour is set by `artifacts/params.json`, which turns the overlay off (`use_reversal_overlay: false`) and adds smoothing (`carry_smooth: 21`). The comment on `reference_nav` in `Config` likewise refers to the overlay's internal cost estimate, which is inactive in the submission. These were left untouched so the frozen commit `d03918c` is exactly what was validated.

---

## 4. Assumptions and known limitations

- **Selection risk.** Carry was discovered on the same development sample used to evaluate it. With all 37 screened signals treated as independent trials, the deflated Sharpe ratio is 0.84. The 95% stationary-bootstrap interval for development Sharpe is [0.75, 2.60].
- **The walk-forward is optimistic.** It is out of sample for construction, not for signal discovery. The smoothing idea was also prompted by 2023–24 evidence, even though the rule was chosen on 2021–22. The full-development Sharpe of 1.66 is the fairer benchmark, and the validation result argues for less than that.
- **Regime dependence.** Both signals were weak in 2021. Validation was a weak-carry period, and its drawdown was outside the development range.
- **Synthetic data.** The economic meaning of `carry_score` is unknown. It is treated as a feature of the data-generating process, not a proven premium.
- **Engine simplifications.** The engine charges no financing or stock-borrow costs, and cash earns zero. A real short book of about 60% of NAV would pay borrow.
- **Execution.** New positions earn only open → close, and costs are priced off the decision-date row, as the engine does. Impact is 5–19% of cost at this scale, and half-spread is the rest.
- **Internal cost estimate.** The strategy cannot observe its NAV, so its internal cost estimate uses a fixed $1m reference. That estimate feeds only the disabled overlay.
- **State and runtime.** State is held within a period and reset automatically when a new period begins, detected from the date going backwards. On the first call, all state is initialised from the supplied history. No files, network, randomness or clock are used. A full development run takes about 5 s, far inside the 60 s per decision and 30 min per period limits.
- **Limited validation evidence.** 132 validation sessions give little power: a Sharpe estimate over six months has a standard error of about 1.4.

---

## 5. Repository map

```
strategy.py                 entry point: generate_positions(history, current_date)
src/strategy_core.py        Allocator, Config, carry targets, band, limits (overlay code kept, disabled)
src/data.py, src/icstats.py panel helpers, engine cost mirror, IC statistics (NW, bootstrap, BH, VR)
artifacts/params.json       frozen submitted configuration
artifacts/phase1/           Phase 1 outputs (integrity, structure, screen, diagnostics, log)
artifacts/phase2/           pre-registration, decision records, validation plan, trial log, all Phase 2 tables
artifacts/validation/       the single validation run at 1x / 1.5x / 2x, plus run_record.json
research/                   every script that produced an artifact (see §1), note and notebook builders
analysis.ipynb              Phase 1 research narrative
research_note.pdf           two-page research note
```

---

## 6. AI disclosure

```text
AI tools used:
- Claude, through the Claude app: the main assistant for the project.

How they were used:
- Claude wrote most of the code in src/, research/, strategy.py and the notebook generator.
```
