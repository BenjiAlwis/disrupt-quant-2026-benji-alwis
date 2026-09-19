"""Phase 2.5 - robustness of the SUBMITTED carry core: every half-year, monthly
consistency, where the P&L comes from, and uncertainty (bootstrap + deflated Sharpe).
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from harness import run, Config, ROOT
sys.path.insert(0, str(ROOT))
from src.icstats import stationary_bootstrap_ci

final = Config(**json.loads((ROOT / "artifacts/params.json").read_text())["config"])
out = ROOT / "artifacts/phase2"

print("=== EVERY HALF-YEAR BLOCK (each starts from cash, liquidates at end) ===")
blocks = [("2021-07-01","2021-12-31","design window"), ("2022-01-01","2022-06-30","design window"),
          ("2022-07-01","2022-12-30","design window"), ("2023-01-01","2023-06-30","OUT-OF-SAMPLE"),
          ("2023-07-01","2023-12-29","OUT-OF-SAMPLE"), ("2024-01-01","2024-06-28","OUT-OF-SAMPLE")]
rows = []
for s, e, tag in blocks:
    r = {"block": f"{s[:7]}..{e[:7]}", "window": tag}
    for mult in (1.0, 1.5, 2.0):
        m, *_ = run(final, s, e, mult, label=f"robust {s} {mult}x", log_trial=False)
        r[f"ret_{mult}x"] = m["annualized_return"]; r[f"sr_{mult}x"] = m["sharpe"]
        if mult == 1.0: r["maxdd"] = m["maximum_drawdown"]
    rows.append(r)
    print(f"{r['block']:18s} {tag:14s} ret {r['ret_1.0x']*100:6.2f}% | SR 1x {r['sr_1.0x']:5.2f}  "
          f"1.5x {r['sr_1.5x']:5.2f}  2x {r['sr_2.0x']:5.2f} | maxDD {r['maxdd']*100:6.2f}%")
pd.DataFrame(rows).to_csv(out / "carry_blocks.csv", index=False)

print("\n=== FULL DEVELOPMENT, ONE PERIOD (2021-01-04 .. 2024-06-28) ===")
m, daily, attr, _ = run(final, "2021-01-04", "2024-06-28", label="carry full dev", log_trial=False)
for k in ("annualized_return","annualized_volatility","sharpe","sortino","maximum_drawdown","calmar",
          "hit_rate","annualized_turnover","transaction_costs","average_gross_exposure",
          "maximum_absolute_net_exposure","constraint_adjustment_days"):
    print(f"  {k:30s} {m[k]:.4f}" if isinstance(m[k], float) else f"  {k:30s} {m[k]}")
daily.to_csv(out / "carry_fulldev_daily.csv", index=False)
attr.to_csv(out / "carry_fulldev_attribution.csv", index=False)

r = daily.set_index(pd.to_datetime(daily.date)).net_return
monthly = (1 + r).resample("ME").prod() - 1
print(f"\n=== MONTHLY CONSISTENCY ===\n  positive months {int((monthly>0).sum())} / {len(monthly)} "
      f"({(monthly>0).mean():.0%})   best {monthly.max()*100:+.2f}%   worst {monthly.min()*100:+.2f}%")
yearly = (1 + r).groupby(r.index.year).prod() - 1
print("  calendar years:", "  ".join(f"{y}: {v*100:+.2f}%" for y, v in yearly.items()))
roll = r.rolling(126).mean() / r.rolling(126).std() * np.sqrt(252)
print(f"  rolling 6-month Sharpe: min {roll.min():.2f}  median {roll.median():.2f}  "
      f"share of windows > 0: {(roll.dropna()>0).mean():.0%}")

print("\n=== WHERE THE P&L COMES FROM ===")
a = attr.copy()
tot = a.net_contribution.sum()
print(f"  long legs  gross {a.long_gross.sum()*100:+.2f}   short legs gross {a.short_gross.sum()*100:+.2f}  "
      f"(sum of daily contributions, % of NAV)")
print(f"  costs {a.cost_contribution.sum()*100:.2f}   net {tot*100:+.2f}")
by_sec = a.groupby("sector").net_contribution.sum().sort_values()
print("  by sector (net, % NAV):", "  ".join(f"{s}: {v*100:+.2f}" for s, v in by_sec.items()))
by_asset = a.groupby("asset_id").net_contribution.sum().sort_values(ascending=False)
top2 = by_asset.head(2).sum() / by_asset[by_asset > 0].sum()
print(f"  assets with positive net contribution: {(by_asset>0).sum()} / 24; "
      f"top two assets = {top2:.0%} of the positive total")
print(f"  max single-asset |weight| ever: {a.target.abs().max():.4f}   "
      f"max sector net: {daily.maximum_sector_net.max():.2e}   max |net|: {daily.net_exposure.abs().max():.2e}")
beta = np.polyfit(daily.market_return, daily.net_return, 1)[0]
print(f"  beta to the equal-weight market: {beta:+.3f}  corr {np.corrcoef(daily.market_return, daily.net_return)[0,1]:+.3f}")

print("\n=== UNCERTAINTY ===")
rv = r.to_numpy(); T = len(rv); sr = rv.mean() / rv.std(ddof=1)
lo, hi = stationary_bootstrap_ci(rv, n_boot=3000, mean_block=10)
boot_sr = []
rng = np.random.default_rng(1)
for _ in range(2000):
    idx = np.empty(T, int); i = rng.integers(T)
    for t in range(T):
        idx[t] = i; i = rng.integers(T) if rng.random() < 0.1 else (i + 1) % T
    x = rv[idx]; boot_sr.append(x.mean() / x.std(ddof=1) * np.sqrt(252))
print(f"  annual Sharpe {sr*np.sqrt(252):.2f}, stationary-bootstrap 95% CI "
      f"[{np.quantile(boot_sr,.025):.2f}, {np.quantile(boot_sr,.975):.2f}]")
g3, g4 = stats.skew(rv), stats.kurtosis(rv, fisher=False)
print(f"  daily skew {g3:+.3f}  kurtosis {g4:.2f}")
print("  Deflated Sharpe ratio (Bailey & Lopez de Prado), null variance 1/T, trials assumed independent:")
gamma = 0.5772156649
for N in (1, 2, 19, 37, 60):
    sr0 = 0.0 if N == 1 else np.sqrt(1 / T) * ((1 - gamma) * stats.norm.ppf(1 - 1 / N)
                                               + gamma * stats.norm.ppf(1 - 1 / (N * np.e)))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    print(f"    N = {N:3d} trials: SR0 = {sr0*np.sqrt(252):.2f} (annual)  DSR = {dsr:.3f}")
