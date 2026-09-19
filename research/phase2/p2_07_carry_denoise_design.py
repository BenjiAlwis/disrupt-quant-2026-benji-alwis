"""Choose a carry de-noising rule on the DESIGN WINDOW ONLY (2021-07 .. 2022-12).
Criterion declared before running: highest Sharpe at 1.5x costs; within 0.05 of the
best, prefer smoothing over refresh (no phase dependence) and the shorter window.
"""
import json
from dataclasses import replace
import pandas as pd
from harness import DEV, Config, ROOT, TRIAL_LOG
from starter.backtester import simulate
from starter.metrics import summarize
from src.strategy_core import Allocator

final = Config(**json.loads((ROOT/"artifacts/params.json").read_text())["config"])
S, E = "2021-07-01", "2022-12-30"
data = DEV[DEV.date <= pd.Timestamp(E)]
def go(cfg, phase=0):
    out = {}
    for mult in (1.0, 1.5, 2.0):
        a = Allocator(cfg); a.session_count = phase
        d, _ = simulate(data, a.decide, start=S, end=E, cost_multiplier=mult)
        out[mult] = summarize(d)
    pd.DataFrame([{"label": f"denoise design refresh={cfg.carry_refresh} smooth={cfg.carry_smooth} phase={phase}",
                   "sharpe": out[1.0]["sharpe"], "window": "design 2021-07..2022-12"}]).to_csv(
        TRIAL_LOG, mode="a", header=False, index=False)
    return out
print(f"DESIGN WINDOW {S} .. {E}")
print(f"{'rule':24s} {'ret':>7s} {'SR 1x':>6s} {'SR 1.5x':>8s} {'SR 2x':>6s} {'turn':>6s} {'maxDD':>7s}")
rows = []
for name, cfg in [("daily (frozen)", final),
                  ("refresh 5", replace(final, carry_refresh=5)),
                  ("refresh 10", replace(final, carry_refresh=10)),
                  ("smooth 5", replace(final, carry_smooth=5)),
                  ("smooth 10", replace(final, carry_smooth=10)),
                  ("smooth 21", replace(final, carry_smooth=21))]:
    o = go(cfg); r = o[1.0]
    rows.append((name, o[1.5]["sharpe"]))
    print(f"{name:24s} {r['annualized_return']*100:6.2f}% {r['sharpe']:6.2f} {o[1.5]['sharpe']:8.2f} "
          f"{o[2.0]['sharpe']:6.2f} {r['annualized_turnover']:6.1f} {r['maximum_drawdown']*100:6.2f}%")
print("\nphase sensitivity of refresh 5 (which day the 5-day cycle starts):")
ph = [go(replace(final, carry_refresh=5), phase=k)[1.5]["sharpe"] for k in range(5)]
print("  SR 1.5x by phase:", " ".join(f"{x:.2f}" for x in ph), f"   range {max(ph)-min(ph):.2f}")
best = max(v for _, v in rows)
eligible = [n for n, v in rows if v >= best - 0.05]
order = ["smooth 5", "smooth 10", "smooth 21", "refresh 5", "refresh 10", "daily (frozen)"]
choice = next(n for n in order if n in eligible)
print(f"\nbest SR 1.5x = {best:.2f}; within 0.05: {eligible}\nRULE SELECTED BY THE DECLARED CRITERION: {choice}")
