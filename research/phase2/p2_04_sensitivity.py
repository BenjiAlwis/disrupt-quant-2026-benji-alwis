"""Phase 2.4 - pre-registered sensitivity grid, one parameter at a time around the
frozen configs, on the out-of-sample period 2023-01 .. 2024-06. REPORTED, NOT USED
TO SELECT: the submitted config is fixed in artifacts/params.json.
"""
import json
from dataclasses import replace
import pandas as pd
from harness import run, Config, ROOT

final = Config(**json.loads((ROOT / "artifacts/params.json").read_text())["config"])
combined = replace(final, use_reversal_overlay=True)
S, E = "2023-01-01", "2024-06-28"
rows = []
def go(base, name, field, values, frozen):
    for v in values:
        cfg = replace(base, **{field: v})
        out = {}
        for mult in (1.0, 2.0):
            m, *_ = run(cfg, S, E, mult, label=f"sens {name} {field}={v} {mult}x")
            out[mult] = m
        rows.append({"model": name, "param": field, "value": v, "frozen": v == frozen,
                     "ret_1x": out[1.0]["annualized_return"], "sharpe_1x": out[1.0]["sharpe"],
                     "sharpe_2x": out[2.0]["sharpe"], "maxdd_1x": out[1.0]["maximum_drawdown"],
                     "turnover": out[1.0]["annualized_turnover"]})
        r = rows[-1]
        print(f"{name:9s} {field:15s} {str(v):>6s}{' *' if r['frozen'] else '  '} "
              f"ret {r['ret_1x']*100:6.2f}%  SR1x {r['sharpe_1x']:5.2f}  SR2x {r['sharpe_2x']:5.2f}  "
              f"maxDD {r['maxdd_1x']*100:6.2f}%  turn {r['turnover']:6.1f}")
print("* = frozen value\n--- submitted model: carry core ---")
go(final, "carry", "carry_gross", [0.8, 1.2, 1.6], 1.2)
go(final, "carry", "carry_band", [0.0, 0.25, 0.5], 0.25)
print("--- documented negative result: carry + overlay ---")
go(combined, "combined", "rev_kappa", [0.75, 1.0, 1.5], 1.0)
go(combined, "combined", "rev_leg", [0.05, 0.10, 0.15], 0.10)
go(combined, "combined", "rev_recalibrate", [10, 21, 63], 21)
pd.DataFrame(rows).to_csv(ROOT / "artifacts/phase2/sensitivity.csv", index=False)
