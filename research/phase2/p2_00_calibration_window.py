"""Phase 2.0 - reversal economics on the CALIBRATION WINDOW ONLY (2021-2022).

Nothing here looks at 2023 or 2024. Design choices for the overlay are made from
this window; the walk-forward test blocks stay unseen until the config is frozen.
"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import Panel, load_split

CAL_END = pd.Timestamp("2022-12-31")
NAV, LEG = 1_000_000.0, 0.10          # reference leg size used to price impact

full = load_split("development")
p = Panel(full[full.date <= CAL_END])
r1 = p("return_1d")
oc = p.forward_open_close(1)
sec = p.sector

# -- 1. reversal coefficient, in return units -----------------------------------
# within-sector demeaned: slope of tomorrow's open->close on today's return.
dm = lambda x: x.sub(x.T.groupby(sec).transform("mean").T)
X, Y = dm(r1).to_numpy().ravel(), dm(oc).to_numpy().ravel()
m = np.isfinite(X) & np.isfinite(Y)
slope = np.polyfit(X[m], Y[m], 1)[0]
b = -slope
# day-clustered SE: one slope per day, then HAC on that daily series
daily = [np.polyfit(dm(r1).iloc[i].values, dm(oc).iloc[i].values, 1)[0]
         for i in range(len(r1) - 1)]
sys.path.insert(0, ".")
from src.icstats import nw_mean_tstat
_, t, _ = nw_mean_tstat(-np.array(daily))
print(f"calibration window {p.dates[0].date()} -> {p.dates[-1].date()} ({len(p.dates)} sessions)")
print(f"reversal coefficient b = {b:.4f}  (fraction of today's sector-relative move that reverses")
print(f"   during tomorrow's session); daily-slope HAC t = {t:.2f}")
for y in (2021, 2022):
    yy = r1.index.year == y
    Xy, Yy = dm(r1)[yy].to_numpy().ravel(), dm(oc)[yy].to_numpy().ravel()
    mm = np.isfinite(Xy) & np.isfinite(Yy)
    print(f"   {y}: b = {-np.polyfit(Xy[mm], Yy[mm], 1)[0]:.4f}")

# -- 2. per-asset one-way cost at the reference leg size ------------------------
dv = (p("volume") * p("close"))
part = LEG * NAV / dv
imp = 0.10 * p("realized_vol_20d") * np.sqrt(part) / np.sqrt(p("liquidity_score").clip(0.1, 1.0))
c = p("spread_bps") / 20_000 + imp                                     # one-way, per date
cm = c.mean().sort_values()
print(f"\none-way cost at a {LEG:.0%} leg (bp), cheapest/dearest:")
print("  " + "  ".join(f"{a}:{v*1e4:.1f}" for a, v in cm.head(6).items()))
print("  " + "  ".join(f"{a}:{v*1e4:.1f}" for a, v in cm.tail(6).items()))

# -- 3. cost-gated within-sector pair overlay (vectorised sketch) ---------------
groups = {s: list(sec[sec == s].index) for s in sec.unique()}
rv, ocv, cv = r1.to_numpy(), oc.to_numpy(), c.to_numpy()
col = {a: j for j, a in enumerate(r1.columns)}
print(f"\npair overlay sketch, leg {LEG:.0%}, best pair per sector, gate: b*|dr| >= kappa * round-trip")
print(f"{'kappa':>6s} {'pairs/day':>10s} {'gross bp/pair':>14s} {'cost bp/pair':>13s} "
      f"{'net bp/pair':>12s} {'sleeve ann %':>13s} {'sleeve Sharpe':>14s}")
for kappa in (1.0, 1.5, 2.0, 3.0):
    pnl = np.zeros(len(r1) - 1); n_pairs = []; g_all = []; c_all = []
    for d in range(len(r1) - 1):
        k = 0
        for s, ids in groups.items():
            best = None
            for a, bb in combinations(ids, 2):
                ia, ib = col[a], col[bb]
                dr = rv[d, ia] - rv[d, ib]
                rt = 2 * (cv[d, ia] + cv[d, ib])
                edge = b * abs(dr) - kappa * rt
                if edge > 0 and (best is None or edge > best[0]):
                    lo, hi = (ib, ia) if dr > 0 else (ia, ib)   # long the loser, short the winner
                    best = (edge, lo, hi, rt)
            if best:
                _, lo, hi, rt = best
                g = LEG * (ocv[d, lo] - ocv[d, hi]); cost = LEG * rt
                pnl[d] += g - cost; g_all.append(g / LEG); c_all.append(rt); k += 1
        n_pairs.append(k)
    sh = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else np.nan
    print(f"{kappa:>6.1f} {np.mean(n_pairs):>10.2f} {np.mean(g_all)*1e4:>14.2f} {np.mean(c_all)*1e4:>13.2f} "
          f"{(np.mean(g_all)-np.mean(c_all))*1e4:>12.2f} {pnl.mean()*252*100:>13.2f} {sh:>14.2f}")
print("\n(sketch ignores netting with carry and assumes a full round trip per pair)")
