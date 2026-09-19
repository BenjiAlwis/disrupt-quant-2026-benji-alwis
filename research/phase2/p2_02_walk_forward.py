"""Phase 2.2 - walk-forward test of the FROZEN config on 2023 H1, 2023 H2, 2024 H1.

The strategy estimates everything it needs from the history supplied at each
decision, so every block is out-of-sample for both design and calibration.
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
from harness import run, fmt, Config, ROOT

pre = json.loads((ROOT / "artifacts/phase2/preregistration.json").read_text())
cfg = Config(**pre["config"])
carry_only = replace(cfg, rev_kappa=1e9)

rows = []
for s, e in pre["test_blocks"]:
    print(f"\n=== TEST BLOCK {s} -> {e} ===")
    for name, c in (("carry only (reference)", carry_only), ("FROZEN: carry + overlay", cfg)):
        for mult in (1.0, 1.5, 2.0):
            m, daily, attr, diag = run(c, s, e, cost_multiplier=mult, label=f"{name} @{mult}x")
            if mult == 1.0 or name.startswith("FROZEN"):
                print(fmt(m))
            rows.append({"block": f"{s}..{e}", "variant": name, "cost_x": mult,
                         "ann_return": m["annualized_return"], "ann_vol": m["annualized_volatility"],
                         "sharpe": m["sharpe"], "max_dd": m["maximum_drawdown"],
                         "turnover": m["annualized_turnover"], "costs": m["transaction_costs"],
                         "pairs_per_day": m["mean_pairs_per_day"], "mean_b": m["mean_b"],
                         "adj_days": m["constraint_adjustment_days"], "hit_rate": m["hit_rate"]})

print("\n=== CONTINUOUS 2023-01-01 -> 2024-06-28 (one period, frozen config) ===")
for mult in (1.0, 1.5, 2.0):
    m, daily, attr, diag = run(cfg, "2023-01-01", "2024-06-28", cost_multiplier=mult,
                               label=f"FROZEN continuous @{mult}x")
    print(fmt(m))
    if mult == 1.0:
        daily.to_csv(ROOT / "artifacts/phase2/walkforward_daily.csv", index=False)
        attr.to_csv(ROOT / "artifacts/phase2/walkforward_attribution.csv", index=False)
        diag.to_csv(ROOT / "artifacts/phase2/walkforward_diagnostics.csv", index=False)
m0, *_ = run(carry_only, "2023-01-01", "2024-06-28", label="carry only continuous @1.0x")
print(fmt(m0))
pd.DataFrame(rows).to_csv(ROOT / "artifacts/phase2/walkforward_blocks.csv", index=False)
