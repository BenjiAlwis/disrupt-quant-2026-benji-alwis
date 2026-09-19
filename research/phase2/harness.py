"""Run the allocator through the OFFICIAL engine over dated blocks.

Every block is a separate simulate() call: it starts from cash, receives all
earlier development rows as history (as the evaluator does), and liquidates at
its last close. Validation data is never loaded here.
"""
from __future__ import annotations
import sys, json
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starter.backtester import simulate, load_data
from starter.metrics import summarize
from src.strategy_core import Allocator, Config

DEV = load_data(ROOT / "data" / "development.csv")
assert DEV.date.max() <= pd.Timestamp("2024-06-28"), "validation must not be loaded"

TRIAL_LOG = ROOT / "artifacts" / "phase2" / "trial_log.csv"


def run(cfg: Config, start: str, end: str, cost_multiplier: float = 1.0, label: str = "",
        log_trial: bool = True):
    alloc = Allocator(cfg)
    data = DEV[DEV.date <= pd.Timestamp(end)]
    daily, attribution = simulate(data, alloc.decide, start=start, end=end,
                                  cost_multiplier=cost_multiplier)
    m = summarize(daily)
    diag = pd.DataFrame(alloc.log)
    m.update({"label": label, "start": start, "end": end, "cost_multiplier": cost_multiplier,
              "mean_pairs_per_day": float(diag.pairs.mean()),
              "mean_b": float(diag.b.mean()),
              "overlay_gross_mean": float(diag.overlay_gross.mean())})
    if log_trial:                     # honest trial count for the deflated-Sharpe discussion
        TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{**{k: v for k, v in m.items() if not isinstance(v, (list, dict))},
                             **{f"cfg_{k}": v for k, v in cfg.as_dict().items()}}])
        row.to_csv(TRIAL_LOG, mode="a", header=not TRIAL_LOG.exists(), index=False)
    return m, daily, attribution, diag


def fmt(m):
    return (f"{m['label']:34s} ret {m['annualized_return']*100:6.2f}%  vol {m['annualized_volatility']*100:5.2f}%  "
            f"Sharpe {m['sharpe']:5.2f}  maxDD {m['maximum_drawdown']*100:6.2f}%  "
            f"turn {m['annualized_turnover']:6.1f}  cost {m['transaction_costs']*100:5.2f}%  "
            f"pairs/d {m['mean_pairs_per_day']:4.2f}  adj {m['constraint_adjustment_days']}")
