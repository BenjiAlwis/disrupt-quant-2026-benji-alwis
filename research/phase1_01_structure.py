"""Phase 1.1 - what kind of market is this?

Establishes the risk structure before any signal search, because the answer
determines what must be hedged rather than predicted:
  * how many priced factors exist (eigenvalues against the Marchenko-Pastur edge)
  * whether sectors are a genuine second block of covariance
  * whether prices trend or mean-revert, with a formal test rather than eyeballing
  * where the daily return actually lives: overnight gap versus intraday session
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Panel, load_split          # noqa: E402
from src.icstats import lo_mackinlay_vr, nw_mean_tstat  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "phase1"


def marchenko_pastur_edge(T: int, N: int) -> tuple[float, float]:
    """Support of the Marchenko-Pastur law for a pure-noise correlation matrix."""
    q = N / T
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2


def run(split: str = "development") -> dict:
    p = Panel(load_split(split))
    ret = p("return_1d")
    close, open_ = p("close"), p("open")
    summary: dict = {"split": split, "n_dates": len(p.dates), "n_assets": len(p.assets)}

    # ---- 1. factor structure -------------------------------------------------
    print("===== FACTOR STRUCTURE =====")
    corr = ret.corr()
    eig = np.linalg.eigvalsh(corr.to_numpy())[::-1]
    T, N = ret.shape
    lo_edge, hi_edge = marchenko_pastur_edge(T, N)
    n_signal = int((eig > hi_edge).sum())
    print(f"  T={T}  N={N}  Marchenko-Pastur support = [{lo_edge:.3f}, {hi_edge:.3f}]")
    print("  eigenvalues:", np.round(eig, 3))
    print(f"  eigenvalues above the noise edge: {n_signal}")
    print("  cumulative variance share:", np.round(np.cumsum(eig) / eig.sum(), 4)[:6])
    summary |= {"eigenvalues": eig.round(6).tolist(), "mp_upper_edge": hi_edge,
                "n_factors_above_noise": n_signal,
                "top_eigen_share": float(eig[0] / eig.sum())}

    # leading eigenvector: is it the market (all loadings same sign)?
    vals, vecs = np.linalg.eigh(corr.to_numpy())
    pc1 = vecs[:, -1]
    pc1 = pc1 * np.sign(pc1.sum())
    print(f"  PC1 loadings all positive: {bool((pc1 > 0).all())}  "
          f"(min {pc1.min():.3f}, max {pc1.max():.3f}) -> a market MODE (common covariance),"
          f"\n     not evidence that the factor carries a risk premium")
    summary["pc1_all_positive"] = bool((pc1 > 0).all())

    # ---- 2. sector block structure -------------------------------------------
    print("\n===== SECTOR BLOCK STRUCTURE =====")
    ids = list(ret.columns)
    within, across = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            same = p.sector[ids[i]] == p.sector[ids[j]]
            (within if same else across).append(corr.iloc[i, j])
    within, across = np.array(within), np.array(across)
    # permutation test: reshuffle sector labels and recompute the gap
    rng = np.random.default_rng(20260918)
    observed_gap = within.mean() - across.mean()
    null = np.empty(5000)
    sec_arr = p.sector.reindex(ids).to_numpy()
    cm = corr.to_numpy()
    iu = np.triu_indices(len(ids), 1)
    pairs = cm[iu]
    for b in range(5000):
        perm = rng.permutation(sec_arr)
        same = (perm[iu[0]] == perm[iu[1]])
        null[b] = pairs[same].mean() - pairs[~same].mean()
    pval = float((np.abs(null) >= abs(observed_gap)).mean())
    print(f"  within-sector mean rho = {within.mean():.4f} (n={within.size})")
    print(f"  across-sector mean rho = {across.mean():.4f} (n={across.size})")
    print(f"  gap = {observed_gap:+.4f}   permutation p = {pval:.4f} (5000 label shuffles)")
    summary |= {"rho_within": float(within.mean()), "rho_across": float(across.mean()),
                "sector_gap": float(observed_gap), "sector_permutation_p": pval}

    # ---- 3. trend or reversion, formally -------------------------------------
    print("\n===== VARIANCE RATIO TESTS (Lo-MacKinlay, heteroskedasticity-robust) =====")
    print("  VR < 1 mean reversion, VR > 1 trending; |z| > 1.96 rejects a random walk")
    mkt = ret.mean(axis=1)
    resid = p.demean_cross_section(ret)
    vr_rows = []
    for label, series_set in (("market factor", {"market": mkt}),
                              ("per asset (median)", {a: ret[a] for a in ret.columns}),
                              ("market-neutral residual (median)", {a: resid[a] for a in resid.columns})):
        for q in (2, 3, 5, 10, 20):
            pairs_ = [lo_mackinlay_vr(s, q) for s in series_set.values()]
            vr = float(np.median([x[0] for x in pairs_]))
            z = float(np.median([x[1] for x in pairs_]))
            vr_rows.append(dict(series=label, q=q, vr=vr, z=z))
            print(f"  {label:34s} q={q:2d}  VR={vr:6.3f}  z={z:+7.2f}")
    pd.DataFrame(vr_rows).to_csv(OUT / f"structure_variance_ratios_{split}.csv", index=False)

    # DESCRIPTIVE ONLY. An earlier version applied a Newey-West HAC estimator to
    # the vector of per-asset autocorrelations. That was wrong: the vector is
    # indexed by asset, not by time, so "lag 1" means "the next asset" and the
    # Bartlett weighting differences sector clustering (78% of adjacent pairs in
    # this ordering share a sector) rather than serial correlation. No HAC
    # statistic is reported here. The inferential evidence for reversal is the
    # daily IC series in phase1_02, which IS indexed by time.
    print("\n===== FIRST-ORDER AUTOCORRELATION (descriptive; see phase1_02 for inference) =====")
    for label, mat in (("raw asset returns", ret), ("market-neutral residual", resid)):
        for lag in (1, 2, 3, 5):
            ac = np.array([mat[a].autocorr(lag) for a in mat.columns])
            print(f"  {label:24s} lag {lag}: mean rho {ac.mean():+.4f}  sd {ac.std(ddof=1):.4f}  "
                  f"range [{ac.min():+.4f}, {ac.max():+.4f}]  share negative {np.mean(ac < 0):.2f}")
    m_ac = [mkt.autocorr(l) for l in (1, 2, 3, 5)]
    print(f"  {'market factor':24s} lags 1-3,5: {np.round(m_ac, 4)}")
    summary["market_autocorr_lag1"] = float(m_ac[0])

    # ---- 4. where does the return live? --------------------------------------
    print("\n===== RETURN DECOMPOSITION: OVERNIGHT GAP vs INTRADAY SESSION =====")
    print("  The engine grants a NEWLY requested position only the open->close move.")
    cc = (close.shift(-1) / close - 1).stack()
    gap = (open_.shift(-1) / close - 1).stack()
    oc = (close.shift(-1) / open_.shift(-1) - 1).stack()
    # These are RATIOS to total variance, not shares of a decomposition: the
    # segments covary (corr 0.77), so Var(cc) = Var(gap) + Var(oc) + 2Cov and
    # the two ratios do not sum to one.
    for name, s in (("close -> close", cc), ("overnight gap", gap), ("open -> close", oc)):
        print(f"  {name:16s} sd = {s.std() * 100:.4f}%   mean = {s.mean() * 100:+.5f}%   "
              f"variance / total variance = {s.var() / cc.var():.3f}")
    cov = float(np.cov(gap, oc)[0, 1])
    print(f"  corr(gap, open->close) = {np.corrcoef(gap, oc)[0, 1]:+.4f}")
    print(f"  ratios sum to {(gap.var() + oc.var()) / cc.var():.3f}; the covariance term "
          f"contributes {2 * cov / cc.var():.3f} -> total {(gap.var() + oc.var() + 2 * cov) / cc.var():.3f}")
    summary |= {"sd_close_close": float(cc.std()), "sd_gap": float(gap.std()),
                "sd_open_close": float(oc.std()),
                "gap_variance_ratio": float(gap.var() / cc.var())}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"structure_summary_{split}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run("development")
