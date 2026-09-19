"""Phase 1.2 - screen every candidate field for tradeable predictive content.

Design decisions that matter:

* Signals are scored against ``open(t+1) -> close(t+1)``, the only part of the
  next session a newly requested position actually earns. The overnight gap is
  reported separately because it accrues only to positions already held.
* Every field is tested raw, market-neutral and sector-neutral. Neutralisation
  is part of the hypothesis, not a cosmetic step: if an edge only survives after
  hedging the market, that is a statement about what must be held flat.
* t-statistics are Newey-West corrected because daily ICs are serially
  correlated.
* 19 fields x 3 neutralisations is a multiple-comparison problem. Benjamini-
  Hochberg FDR control at q = 0.10 is applied across the whole screen and the
  surviving set is reported, so that a field is not promoted on a raw p-value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Panel, load_split                                    # noqa: E402
from src.icstats import (benjamini_hochberg, daily_ic, summarise_ic,      # noqa: E402
                         tercile_spread, nw_mean_tstat)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "phase1"

# Every observable field that could plausibly rank assets cross-sectionally.
# Fields constant within a sector are included deliberately so the degeneracy
# guard is exercised rather than silently producing NaN.
CANDIDATES = [
    "return_1d", "return_5d", "return_20d", "price_strength_20", "price_strength_60",
    "realized_vol_5d", "realized_vol_20d", "volume_zscore", "volume_trend",
    "spread_bps", "liquidity_score", "sentiment_score", "flow_index", "carry_score",
    "event_intensity", "event_flag", "intraday_range", "volume", "sector_dispersion",
]
# "market_neutral" is deliberately NOT a separate hypothesis. Subtracting the
# daily cross-sectional mean adds the same constant to every asset that day, and
# a within-day correlation -- Spearman or Pearson -- is invariant to it. The raw
# and market-neutral ICs are therefore identical to machine precision, and
# counting them separately would inflate the multiple-testing correction with
# duplicates. It is asserted as an invariance check instead (see check_invariance).
NEUTRALISATIONS = ("raw", "sector_neutral")


def neutralise(panel: Panel, wide: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "raw":
        return wide
    if mode == "market_neutral":
        return panel.demean_cross_section(wide)
    if mode == "sector_neutral":
        return panel.demean_sector(wide)
    raise ValueError(mode)


def check_market_neutral_invariance(panel: Panel, field: str = "return_1d") -> float:
    """Assert that market-neutralisation cannot change a within-day correlation.

    Returns the largest absolute difference between the raw and market-neutral
    daily IC series, which must be at machine-precision level.
    """
    fwd = panel.forward_open_close(1)
    raw = daily_ic(panel(field), fwd)
    neutral = daily_ic(panel.demean_cross_section(panel(field)),
                       panel.demean_cross_section(fwd))
    return float((raw - neutral).abs().max())


def run(split: str = "development", n_boot: int = 1000) -> pd.DataFrame:
    p = Panel(load_split(split))
    fwd_oc = p.forward_open_close(1)     # tradeable for a new position
    fwd_cc = p.forward_close_close(1)    # reference only
    fwd_gap = p.overnight_gap()          # accrues to existing holdings only

    gap = check_market_neutral_invariance(p)
    print(f"invariance check: max |IC_raw - IC_market_neutral| = {gap:.2e}")
    print("market-neutralisation cannot change a within-day correlation, so it is")
    print("an invariance check rather than a separate hypothesis in the screen.\n")

    records = []
    for field in CANDIDATES:
        raw = p(field)
        for mode in NEUTRALISATIONS:
            sig = neutralise(p, raw, mode)
            fwd = neutralise(p, fwd_oc, mode)
            if p.is_degenerate(sig):
                records.append(dict(field=field, neutralisation=mode, degenerate=True,
                                    n_days=0, mean_ic=np.nan, t_naive=np.nan, t_nw=np.nan,
                                    p_nw=np.nan, ci_low=np.nan, ci_high=np.nan,
                                    hit_rate=np.nan, ir_annual=np.nan))
                continue
            res = summarise_ic(daily_ic(sig, fwd), n_boot=n_boot)
            rec = dict(field=field, neutralisation=mode, degenerate=False) | res.as_dict()
            records.append(rec)

    table = pd.DataFrame(records)
    live = ~table.degenerate
    reject, cutoff = benjamini_hochberg(table.loc[live, "p_nw"].to_numpy(), q=0.10)
    table["bh_significant"] = False
    table.loc[live, "bh_significant"] = reject
    table["abs_t"] = table.t_nw.abs()

    print(f"===== CROSS-SECTIONAL IC SCREEN ({split}) =====")
    print(f"  scored against open(t+1)->close(t+1); {int(live.sum())} live tests; "
          f"BH q=0.10 cutoff p <= {cutoff:.5f}")
    print(f"  {'field':18s} {'neutral':15s} {'IC':>8s} {'t_nw':>7s} {'t_naive':>8s} "
          f"{'p_nw':>8s} {'95% CI':>18s} {'BH':>4s}")
    for _, r in table.sort_values("abs_t", ascending=False).iterrows():
        if r.degenerate:
            continue
        ci = f"[{r.ci_low:+.4f},{r.ci_high:+.4f}]"
        print(f"  {r.field:18s} {r.neutralisation:15s} {r.mean_ic:>8.4f} {r.t_nw:>7.2f} "
              f"{r.t_naive:>8.2f} {r.p_nw:>8.4f} {ci:>18s} {'YES' if r.bh_significant else '':>4s}")

    degen = table.loc[table.degenerate, ["field", "neutralisation"]]
    if len(degen):
        print("\n  excluded as degenerate (no cross-sectional variation after neutralisation):")
        for _, r in degen.iterrows():
            print(f"    {r.field} / {r.neutralisation}")

    # ---- return-segment decomposition for the survivors ----------------------
    survivors = (table[table.bh_significant]
                 .sort_values("abs_t", ascending=False)
                 .drop_duplicates("field").field.tolist())
    print(f"\n===== SEGMENT DECOMPOSITION FOR BH SURVIVORS ({split}) =====")
    print("  Does the edge sit in the tradeable session or in the untradeable gap?")
    print(f"  {'field':18s} {'segment':16s} {'IC':>8s} {'t_nw':>7s}")
    seg_rows = []
    for field in survivors:
        sig = p.demean_sector(p(field))
        for name, fwd in (("close->close", fwd_cc), ("overnight gap", fwd_gap),
                          ("open->close", fwd_oc)):
            res = summarise_ic(daily_ic(sig, p.demean_sector(fwd)), n_boot=n_boot)
            print(f"  {field:18s} {name:16s} {res.mean_ic:>8.4f} {res.t_nw:>7.2f}")
            seg_rows.append(dict(field=field, segment=name, mean_ic=res.mean_ic, t_nw=res.t_nw))
        print()

    # ---- bucket monotonicity -------------------------------------------------
    print(f"===== TERCILE SPREADS FOR BH SURVIVORS ({split}) =====")
    print("  Equal-weight sort on the sector-neutral signal; returns in annualised %.")
    print(f"  {'field':18s} {'low':>9s} {'mid':>9s} {'high':>9s} {'spread':>9s} "
          f"{'t_nw':>7s} {'monotone':>9s}")
    bucket_rows = []
    for field in survivors:
        sig = p.demean_sector(p(field))
        buckets = tercile_spread(sig, p.demean_sector(fwd_oc))
        ann = buckets.mean() * 252 * 100
        _, t_nw, _ = nw_mean_tstat(buckets.spread)
        mono = bool(ann.q1 < ann.q2 < ann.q3 or ann.q1 > ann.q2 > ann.q3)
        print(f"  {field:18s} {ann.q1:>9.2f} {ann.q2:>9.2f} {ann.q3:>9.2f} "
              f"{ann.spread:>9.2f} {t_nw:>7.2f} {str(mono):>9s}")
        bucket_rows.append(dict(field=field, q1=ann.q1, q2=ann.q2, q3=ann.q3,
                                spread_annual_pct=ann.spread, t_nw=t_nw, monotone=mono))

    OUT.mkdir(parents=True, exist_ok=True)
    table.drop(columns="abs_t").to_csv(OUT / f"screen_ic_{split}.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(OUT / f"screen_segments_{split}.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(OUT / f"screen_terciles_{split}.csv", index=False)
    (OUT / f"screen_survivors_{split}.json").write_text(
        json.dumps({"bh_cutoff": cutoff, "survivors": survivors}, indent=2))
    return table


if __name__ == "__main__":
    run("development")
