"""Compare an externally supplied candidate against the frozen strategy on DEVELOPMENT only,
same engine, same blocks. Each run loads a fresh copy of the candidate module (it keeps
module-level state)."""
import importlib.util, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from harness import DEV, Config, ROOT, run
from starter.backtester import simulate
from starter.metrics import summarize

def fresh_candidate(overlay=True):
    spec = importlib.util.spec_from_file_location(f"cand{time.time_ns()}", ROOT/"research/phase2/candidate_external.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not overlay: mod.MIN_CALIBRATION_SESSIONS = 10**9     # carry part alone
    return mod

def run_cand(s, e, mult=1.0, overlay=True):
    mod = fresh_candidate(overlay)
    d, _ = simulate(DEV[DEV.date <= pd.Timestamp(e)], mod.generate_positions, start=s, end=e, cost_multiplier=mult)
    return summarize(d), d

final = Config(**json.loads((ROOT/"artifacts/params.json").read_text())["config"])
combined = replace(final, use_reversal_overlay=True)
periods = [("2023-01-01","2023-06-30"),("2023-07-01","2023-12-29"),("2024-01-01","2024-06-28"),
           ("2023-01-01","2024-06-28"),("2021-07-01","2024-06-28")]
print(f"{'period':24s} {'model':26s} {'ret':>7s} {'SR 1x':>6s} {'SR 1.5x':>7s} {'SR 2x':>6s} {'maxDD':>7s} {'turn':>6s} {'adj':>4s}")
rows=[]
for s,e in periods:
    for name, fn in (("candidate (full)", lambda m: run_cand(s,e,m,True)),
                     ("candidate carry part only", lambda m: run_cand(s,e,m,False)),
                     ("FROZEN carry core", lambda m: run(final,s,e,m,log_trial=False)[:2]),
                     ("our combined (rejected)", lambda m: run(combined,s,e,m,log_trial=False)[:2])):
        res = {m: fn(m)[0] for m in (1.0,1.5,2.0)}
        r=res[1.0]
        print(f"{s[:7]}..{e[:7]:14s} {name:26s} {r['annualized_return']*100:6.2f}% {r['sharpe']:6.2f} "
              f"{res[1.5]['sharpe']:7.2f} {res[2.0]['sharpe']:6.2f} {r['maximum_drawdown']*100:6.2f}% "
              f"{r['annualized_turnover']:6.1f} {r['constraint_adjustment_days']:4d}")
        rows.append({"period":f"{s}..{e}","model":name,"ret":r['annualized_return'],"sr1":r['sharpe'],
                     "sr15":res[1.5]['sharpe'],"sr2":res[2.0]['sharpe'],"maxdd":r['maximum_drawdown'],
                     "turnover":r['annualized_turnover']})
    print()
pd.DataFrame(rows).to_csv(ROOT/"artifacts/phase2/candidate_comparison.csv",index=False)
