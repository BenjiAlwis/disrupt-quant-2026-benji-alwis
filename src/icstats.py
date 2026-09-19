"""Inference tools for daily cross-sectional statistics.

Daily information coefficients are serially correlated and are computed for many
candidate fields at once. Both facts inflate naive significance, so this module
provides Newey-West corrected t-statistics, a stationary bootstrap for interval
estimates, and Benjamini-Hochberg false-discovery-rate control.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Serial-correlation robust inference
# ---------------------------------------------------------------------------

def newey_west_lag(n: int) -> int:
    """Newey-West (1994) automatic bandwidth: floor(4 * (n/100)^(2/9))."""
    return max(0, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def nw_mean_tstat(x: np.ndarray | pd.Series, lag: int | None = None) -> tuple[float, float, int]:
    """Mean, HAC t-statistic for H0: mean == 0, and the bandwidth used.

    Uses the Bartlett-kernel long-run variance of the sample mean.
    """
    x = pd.Series(x).dropna().to_numpy(float)
    n = x.size
    if n < 3:
        return (float(x.mean()) if n else np.nan), np.nan, 0
    lag = newey_west_lag(n) if lag is None else lag
    dev = x - x.mean()
    gamma0 = float(dev @ dev) / n
    lrv = gamma0
    for j in range(1, lag + 1):
        gamma_j = float(dev[j:] @ dev[:-j]) / n
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * gamma_j
    if lrv <= 0:                      # Bartlett weights keep this psd in practice
        lrv = gamma0                  # fall back rather than emit a complex se
    se = np.sqrt(lrv / n)
    return float(x.mean()), float(x.mean() / se) if se > 0 else np.nan, lag


def stationary_bootstrap_ci(x: np.ndarray | pd.Series, n_boot: int = 2000,
                            mean_block: float = 10.0, alpha: float = 0.05,
                            seed: int = 20260918) -> tuple[float, float]:
    """Politis-Romano stationary bootstrap confidence interval for the mean.

    Geometric block lengths with expectation ``mean_block`` preserve short-range
    dependence without imposing a fixed block boundary.
    """
    x = pd.Series(x).dropna().to_numpy(float)
    n = x.size
    if n < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(0, n) if rng.random() < p else (i + 1) % n
        means[b] = x[idx].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, float]:
    """BH step-up FDR control. Returns the rejection mask and the p-value cutoff."""
    p = np.asarray(pvalues, float)
    finite = np.isfinite(p)
    order = np.argsort(np.where(finite, p, np.inf))
    m = int(finite.sum())
    reject = np.zeros_like(p, dtype=bool)
    if m == 0:
        return reject, 0.0
    ranked = p[order][:m]
    thresholds = q * np.arange(1, m + 1) / m
    passing = np.where(ranked <= thresholds)[0]
    if passing.size == 0:
        return reject, 0.0
    kmax = passing[-1]
    reject[order[: kmax + 1]] = True
    return reject, float(thresholds[kmax])


# ---------------------------------------------------------------------------
# Cross-sectional information coefficients
# ---------------------------------------------------------------------------

@dataclass
class ICResult:
    n_days: int
    mean_ic: float
    t_naive: float
    t_nw: float
    nw_lag: int
    p_nw: float
    ci_low: float
    ci_high: float
    hit_rate: float
    ir_annual: float

    def as_dict(self) -> dict:
        return asdict(self)


def daily_ic(signal: pd.DataFrame, forward: pd.DataFrame, method: str = "spearman",
             min_assets: int = 8) -> pd.Series:
    """Per-date cross-sectional correlation between a signal and a forward return.

    ``signal`` must be observable at the close of the index date; ``forward``
    must hold the return earned strictly after it. Alignment is on the shared
    index, so the caller is responsible for the shift convention.
    """
    sig, fwd = signal.align(forward, join="inner")
    out: dict[pd.Timestamp, float] = {}
    sig_v, fwd_v = sig.to_numpy(float), fwd.to_numpy(float)
    for i, date in enumerate(sig.index):
        a, b = sig_v[i], fwd_v[i]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < min_assets:
            continue
        a, b = a[mask], b[mask]
        if np.std(a) <= 1e-14 or np.std(b) <= 1e-14:
            continue
        out[date] = (stats.spearmanr(a, b).statistic if method == "spearman"
                     else float(np.corrcoef(a, b)[0, 1]))
    return pd.Series(out, dtype=float).sort_index()


def summarise_ic(ic: pd.Series, n_boot: int = 2000) -> ICResult:
    """Full inference summary for a daily IC series."""
    ic = ic.dropna()
    n = len(ic)
    if n < 3:
        return ICResult(n, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan, np.nan)
    mean, t_nw, lag = nw_mean_tstat(ic)
    t_naive = float(mean / (ic.std(ddof=1) / np.sqrt(n)))
    p_nw = float(2 * stats.norm.sf(abs(t_nw))) if np.isfinite(t_nw) else np.nan
    lo, hi = stationary_bootstrap_ci(ic, n_boot=n_boot)
    return ICResult(
        n_days=n, mean_ic=float(mean), t_naive=t_naive, t_nw=float(t_nw), nw_lag=lag,
        p_nw=p_nw, ci_low=lo, ci_high=hi, hit_rate=float((ic > 0).mean()),
        ir_annual=float(mean / ic.std(ddof=1) * np.sqrt(252)) if ic.std(ddof=1) > 0 else np.nan,
    )


# ---------------------------------------------------------------------------
# Portfolio-level evidence
# ---------------------------------------------------------------------------

def tercile_spread(signal: pd.DataFrame, forward: pd.DataFrame, n_buckets: int = 3,
                   min_assets: int = 12) -> pd.DataFrame:
    """Equal-weight bucket returns from sorting on the signal each date.

    An IC says the ranking is informative; bucket monotonicity says the payoff is
    not driven by a single tail. With 24 assets, terciles give 8 names a bucket.
    """
    sig, fwd = signal.align(forward, join="inner")
    rows = []
    sig_v, fwd_v = sig.to_numpy(float), fwd.to_numpy(float)
    for i, date in enumerate(sig.index):
        a, b = sig_v[i], fwd_v[i]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < min_assets or np.std(a[mask]) <= 1e-14:
            continue
        a, b = a[mask], b[mask]
        order = np.argsort(a, kind="stable")
        edges = np.linspace(0, a.size, n_buckets + 1).round().astype(int)
        row = {"date": date}
        for k in range(n_buckets):
            row[f"q{k + 1}"] = float(b[order[edges[k]:edges[k + 1]]].mean())
        row["spread"] = row[f"q{n_buckets}"] - row["q1"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def lo_mackinlay_vr(series: pd.Series, q: int) -> tuple[float, float]:
    """Variance ratio and heteroskedasticity-robust z-statistic (Lo-MacKinlay 1988).

    VR < 1 indicates mean reversion, VR > 1 trending, VR == 1 a random walk.
    """
    x = np.log1p(pd.Series(series).dropna().to_numpy(float))
    n = x.size
    if n <= q + 1:
        return np.nan, np.nan
    mu = x.mean()
    dev = x - mu
    var1 = float(dev @ dev) / (n - 1)
    agg = pd.Series(x).rolling(q).sum().dropna().to_numpy()
    varq = float(np.var(agg, ddof=1)) / q
    vr = varq / var1 if var1 > 0 else np.nan
    # Lo-MacKinlay heteroskedasticity-consistent variance of the VR statistic.
    # delta_j already carries the 1/n scaling through the ratio of sums, so the
    # standardised statistic is (VR - 1) / sqrt(theta) with no extra sqrt(n).
    denom = float(dev @ dev) ** 2
    theta = 0.0
    for j in range(1, q):
        d = float(dev[j:] ** 2 @ dev[:-j] ** 2)
        delta_j = d / denom if denom > 0 else np.nan
        theta += (2.0 * (q - j) / q) ** 2 * delta_j
    z = (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
    return float(vr), float(z)
