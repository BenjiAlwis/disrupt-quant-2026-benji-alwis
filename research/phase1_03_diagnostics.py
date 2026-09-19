"""Phase 1.3 - decay, stability, conditioning, orthogonality and the cost bridge.

The screen establishes that two fields rank next-session returns. This module
asks the four questions that decide whether either is worth trading:

  1. How fast does the edge decay? Decay sets the required trading frequency.
  2. Is it stable across sub-periods, or an artefact of one regime?
  3. Is it conditionally stronger in states that are also cheap to trade?
  4. What survives the actual cost function at realistic portfolio size?

Question 4 is the one that kills most short-horizon signals, so it is answered
with the engine's own cost arithmetic rather than a flat basis-point assumption.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Panel, load_split, one_way_cost_rate     # noqa: E402
from src.icstats import daily_ic, summarise_ic, nw_mean_tstat  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "phase1"
INITIAL_NAV = 1_000_000.0

SIGNALS = {"reversal": lambda p: -p("return_1d"), "carry": lambda p: p("carry_score")}


def rank_weights(sig: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Cross-sectional rank -> dollar-neutral weights with fixed gross exposure.

    A deliberately naive construction: it isolates the signal's raw economics
    without any cost optimisation, so the cost bridge below measures the problem
    rather than a solution to it.
    """
    ranks = sig.rank(axis=1)
    centred = ranks.sub(ranks.mean(axis=1), axis=0)
    scale = centred.abs().sum(axis=1).replace(0, np.nan)
    return centred.div(scale, axis=0) * gross


def run(split: str = "development", n_boot: int = 1000) -> None:
    p = Panel(load_split(split))
    frame = load_split(split)
    fwd_oc = p.forward_open_close(1)
    sn = p.demean_sector

    # ---- 1. decay profile ----------------------------------------------------
    print(f"===== SIGNAL DECAY ({split}) =====")
    print("  IC against open(t+k) -> close(t+k), sector-neutral. Decay sets trade frequency.")
    print(f"  {'k':>3} " + " ".join(f"{n + ' IC':>12s} {'t':>6s}" for n in SIGNALS))
    decay_rows = []
    for k in (1, 2, 3, 4, 5, 10, 20):
        fwd_k = sn(p.forward_open_close(k))
        line = f"  {k:>3} "
        for name, fn in SIGNALS.items():
            res = summarise_ic(daily_ic(sn(fn(p)), fwd_k), n_boot=200)
            line += f" {res.mean_ic:>12.4f} {res.t_nw:>6.2f}"
            decay_rows.append(dict(signal=name, k=k, mean_ic=res.mean_ic, t_nw=res.t_nw))
        print(line)

    # ---- 2. stability --------------------------------------------------------
    ic_series = {n: daily_ic(sn(f(p)), sn(fwd_oc)) for n, f in SIGNALS.items()}
    print(f"\n===== SUB-PERIOD STABILITY ({split}) =====")
    print(f"  {'period':>14s} {'n':>5s} " + " ".join(f"{n + ' IC':>12s} {'t_nw':>6s}" for n in SIGNALS))
    stab_rows = []
    ref = next(iter(ic_series.values()))
    years = pd.Series(ref.index).dt.year.to_numpy()
    blocks = [(str(y), (years == y)) for y in sorted(set(years))]
    n = len(ref)
    blocks += [("first half", np.arange(n) < n // 2), ("second half", np.arange(n) >= n // 2)]
    for label, mask in blocks:
        line = f"  {label:>14s} {int(mask.sum()):>5d} "
        for name, ic in ic_series.items():
            sub = ic[mask]
            _, t, _ = nw_mean_tstat(sub)
            line += f" {sub.mean():>12.4f} {t:>6.2f}"
            stab_rows.append(dict(signal=name, period=label, n=int(mask.sum()),
                                  mean_ic=float(sub.mean()), t_nw=float(t)))
        print(line)
    for name, ic in ic_series.items():
        print(f"  {name}: share of days with positive IC = {(ic > 0).mean():.3f}")
    keys = list(ic_series)
    print(f"  corr(IC_{keys[0]}, IC_{keys[1]}) = {ic_series[keys[0]].corr(ic_series[keys[1]]):+.3f}"
          "   -> diversification between sleeves")

    # ---- 3. conditioning -----------------------------------------------------
    print(f"\n===== CONDITIONAL STRENGTH ({split}) =====")
    print("  Median split each day; reversal IC computed within each half.")
    print(f"  {'conditioner':18s} {'low half':>10s} {'t':>6s} {'high half':>10s} {'t':>6s} {'diff':>8s}")
    cond_rows = []
    sig_rev, fwd_n = sn(SIGNALS["reversal"](p)), sn(fwd_oc)
    sv, fv = sig_rev.to_numpy(float), fwd_n.to_numpy(float)
    for cname in ("realized_vol_20d", "liquidity_score", "volume_zscore", "spread_bps",
                  "event_intensity", "market_volatility"):
        cv = p(cname).reindex(sig_rev.index).to_numpy(float)
        lo, hi = [], []
        for i in range(len(sig_rev)):
            a, b, c = sv[i], fv[i], cv[i]
            m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            if m.sum() < 12 or np.nanstd(c[m]) <= 1e-14:
                continue
            med = np.median(c[m])
            for mask, dest in (((c <= med) & m, lo), ((c > med) & m, hi)):
                if mask.sum() >= 6 and np.std(a[mask]) > 1e-14 and np.std(b[mask]) > 1e-14:
                    dest.append(stats.spearmanr(a[mask], b[mask]).statistic)
        if not lo or not hi:
            print(f"  {cname:18s} {'constant cross-sectionally - skipped':>44s}")
            continue
        lo, hi = pd.Series(lo), pd.Series(hi)
        _, tl, _ = nw_mean_tstat(lo)
        _, th, _ = nw_mean_tstat(hi)
        print(f"  {cname:18s} {lo.mean():>10.4f} {tl:>6.2f} {hi.mean():>10.4f} {th:>6.2f} "
              f"{hi.mean() - lo.mean():>+8.4f}")
        cond_rows.append(dict(conditioner=cname, ic_low=float(lo.mean()), t_low=float(tl),
                              ic_high=float(hi.mean()), t_high=float(th),
                              diff=float(hi.mean() - lo.mean())))

    # ---- 4. incremental content ---------------------------------------------
    print(f"\n===== ORTHOGONALITY ({split}) =====")
    print("  Daily cross-sectional regression of one signal's rank on the other's.")
    rev_r = sn(SIGNALS["reversal"](p)).rank(axis=1)
    car_r = sn(SIGNALS["carry"](p)).rank(axis=1)
    raw_corr = np.nanmean([stats.spearmanr(rev_r.iloc[i], car_r.iloc[i]).statistic
                           for i in range(len(rev_r))])
    print(f"  mean daily cross-sectional rank corr(reversal, carry) = {raw_corr:+.4f}")
    for target, other, label in ((car_r, rev_r, "carry | reversal"), (rev_r, car_r, "reversal | carry")):
        resid = {}
        for i in range(len(target)):
            y, x = target.iloc[i].to_numpy(float), other.iloc[i].to_numpy(float)
            m = np.isfinite(y) & np.isfinite(x)
            if m.sum() < 8 or np.std(x[m]) <= 1e-14:
                continue
            beta = np.polyfit(x[m], y[m], 1)
            r = np.full(y.shape, np.nan)
            r[m] = y[m] - np.polyval(beta, x[m])
            resid[target.index[i]] = r
        rdf = pd.DataFrame(resid, index=target.columns).T
        res = summarise_ic(daily_ic(rdf, sn(fwd_oc)), n_boot=n_boot)
        print(f"  residual IC of {label:20s} = {res.mean_ic:+.4f}  t_nw {res.t_nw:+.2f}")

    # ---- 5. cost bridge ------------------------------------------------------
    print(f"\n===== COST BRIDGE ({split}) =====")
    print(f"  Naive daily rebalance to rank weights at NAV ${INITIAL_NAV:,.0f}, "
          "priced with the engine's own cost model.")
    print(f"  {'signal':10s} {'gross':>7s} {'gross α':>9s} {'turnover':>9s} {'cost':>9s} "
          f"{'net α':>9s} {'net/gross':>10s} {'breakeven':>10s}")
    cost_rows = []
    by_date = {d: g.reset_index(drop=True) for d, g in frame.groupby("date")}
    dates = list(p.dates)
    for name, fn in SIGNALS.items():
        for gross in (0.5, 1.0, 2.0):
            w = rank_weights(sn(fn(p)), gross=gross)
            prev = np.zeros(len(p.assets))
            g_ret, costs, turns = [], [], []
            for i in range(len(dates) - 1):
                d = dates[i]
                tgt = w.loc[d].to_numpy(float)
                if not np.isfinite(tgt).all():
                    continue
                delta = tgt - prev
                rate = one_way_cost_rate(by_date[d], delta, INITIAL_NAV)
                costs.append(float(np.abs(delta) @ rate))
                turns.append(float(np.abs(delta).sum()))
                r = fwd_oc.loc[d].to_numpy(float)
                g_ret.append(float(np.nansum(tgt * r)))
                prev = tgt
            g_ret, costs, turns = np.array(g_ret), np.array(costs), np.array(turns)
            ga, ca = g_ret.mean() * 252, costs.mean() * 252
            net = g_ret - costs
            sharpe = net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) > 0 else np.nan
            be = ga / ca if ca > 0 else np.inf
            print(f"  {name:10s} {gross:>7.1f} {ga * 100:>8.2f}% {turns.mean():>9.3f} "
                  f"{ca * 100:>8.2f}% {(ga - ca) * 100:>8.2f}% {(ga - ca) / ga:>10.2f} {be:>10.2f}")
            cost_rows.append(dict(signal=name, gross=gross, gross_alpha=ga, turnover=turns.mean(),
                                  cost=ca, net_alpha=ga - ca, net_sharpe=sharpe,
                                  cost_multiple_to_breakeven=be))
    print("\n  'breakeven' = the cost multiplier at which net alpha reaches zero.")
    print("  The engine stress-tests costs at 1.0x, 1.5x and 2.0x.")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decay_rows).to_csv(OUT / f"diag_decay_{split}.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(OUT / f"diag_stability_{split}.csv", index=False)
    pd.DataFrame(cond_rows).to_csv(OUT / f"diag_conditioning_{split}.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / f"diag_cost_bridge_{split}.csv", index=False)
    pd.DataFrame(ic_series).to_csv(OUT / f"diag_ic_series_{split}.csv")


if __name__ == "__main__":
    run("development")
