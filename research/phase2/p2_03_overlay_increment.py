"""Phase 2.3 - is the overlay's out-of-sample contribution distinguishable from zero?
Both models were fully specified before the test blocks were run (carry-only was the
pre-registered reference), so comparing them here is a two-model comparison, not tuning.
"""
import json, sys
from dataclasses import replace
import numpy as np, pandas as pd
from harness import run, Config, ROOT
sys.path.insert(0, str(ROOT))
from src.icstats import nw_mean_tstat, stationary_bootstrap_ci

pre = json.loads((ROOT / "artifacts/phase2/preregistration.json").read_text())
cfg = Config(**pre["config"]); carry = replace(cfg, rev_kappa=1e9)
S, E = "2023-01-01", "2024-06-28"
print(f"{'cost':>5s} {'combined':>22s} {'carry only':>22s} {'overlay adds':>14s}")
res = {}
for mult in (1.0, 1.5, 2.0):
    mc, dc, *_ = run(cfg, S, E, mult, label=f"inc combined {mult}x")
    m0, d0, *_ = run(carry, S, E, mult, label=f"inc carry {mult}x")
    res[mult] = (dc, d0)
    print(f"{mult:>4.1f}x  ret {mc['annualized_return']*100:5.2f}% SR {mc['sharpe']:4.2f}   "
          f"ret {m0['annualized_return']*100:5.2f}% SR {m0['sharpe']:4.2f}   "
          f"{(mc['annualized_return']-m0['annualized_return'])*100:+6.2f}pp")
dc, d0 = res[1.0]
inc = dc.net_return.to_numpy() - d0.net_return.to_numpy()
mean, t, _ = nw_mean_tstat(inc)
lo, hi = stationary_bootstrap_ci(inc, n_boot=2000, mean_block=10)
print(f"\ndaily overlay increment at 1x: mean {mean*252*100:+.2f}%/yr  HAC t {t:+.2f}  "
      f"95% CI [{lo*252*100:+.2f}, {hi*252*100:+.2f}] %/yr")
print(f"corr(combined, carry-only daily returns) = {np.corrcoef(dc.net_return, d0.net_return)[0,1]:.3f}")
diag = pd.read_csv(ROOT / "artifacts/phase2/walkforward_diagnostics.csv", parse_dates=["date"])
print(f"\nb used out-of-sample: {diag.b.min():.4f} .. {diag.b.max():.4f} (calibration-window estimate was 0.0381)")
for y, g in diag.groupby(diag.date.dt.to_period('Q')):
    print(f"   {y}: mean b {g.b.mean():.4f}  pairs/day {g.pairs.mean():.2f}")
