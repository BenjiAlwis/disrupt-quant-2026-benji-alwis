"""Descriptive diagnostics for the single public-validation run. Runs no strategy and
changes nothing: it reads the committed validation output and development results.

    python research/validation_diagnostics.py

Reproduces the Section 6 figures and the assessment in artifacts/validation/run_record.json:
  * traded carry IC on validation (21-session mean, sector-neutral, vs next open->close)
  * how often an overlapping 132-session development window was at least as weak
  * the worst 132-session development drawdown, for comparison with validation's
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import Panel, load_combined          # noqa: E402
from src.icstats import daily_ic, nw_mean_tstat    # noqa: E402

VAL_START = pd.Timestamp("2024-07-01")
val = json.loads((ROOT / "artifacts/validation/metrics_x1.0.json").read_text())
n = val["sessions"]

# ---- signal: did carry reverse, or just weaken? ----------------------------------------
p = Panel(load_combined())                        # development rows supply the 21-day history
fwd = p.demean_sector(p.forward_open_close())
ic = daily_ic(p.demean_sector(p("carry_score").rolling(21).mean()), fwd)
mean, t, lag = nw_mean_tstat(ic[ic.index >= VAL_START])
print(f"validation carry IC {mean:+.3f}  (Newey-West t {t:.2f}, L={lag})")

# ---- where does validation sit among development windows of the same length? -----------
r = pd.read_csv(ROOT / "artifacts/phase2/carry_fulldev_daily.csv").net_return
sharpe = r.rolling(n).apply(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)).dropna()


def max_dd(x):
    nav = np.cumprod(1 + x)
    return float((nav / np.maximum.accumulate(nav) - 1).min())


dd = r.rolling(n).apply(max_dd, raw=True).dropna()
print(f"{len(sharpe)} overlapping {n}-session development windows:")
print(f"  Sharpe <= validation ({val['sharpe']:.2f}): {(sharpe <= val['sharpe']).mean():.1%}; "
      f"worst window {sharpe.min():.2f}")
print(f"  worst window drawdown {dd.min():.2%} vs validation {val['maximum_drawdown']:.2%}")
