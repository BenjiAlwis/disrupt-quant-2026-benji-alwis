"""Phase 1.0 - data integrity and look-ahead audit.

Two questions:
  1. Does the data honour the invariants the dictionary claims (price ordering,
     quote ordering, stated field definitions)?
  2. Is every derived field computable from information available at or before
     its own date? A field that silently embedded a future observation would be
     the most dangerous thing in this dataset - it would look like enormous
     alpha in research and vanish under evaluation.

Every derived field is recomputed from raw prices/volumes using only data up to
and including date t, then compared to the supplied column.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Panel, load_split  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "phase1"


def audit(split: str = "development") -> pd.DataFrame:
    frame = load_split(split)
    p = Panel(frame)
    close, open_, high, low = p("close"), p("open"), p("high"), p("low")
    volume, bid, ask = p("volume"), p("bid"), p("ask")
    ret1 = close / close.shift(1) - 1.0

    print(f"===== INVARIANTS ({split}) =====")
    checks = {
        "high >= max(open, close)": bool((high >= np.maximum(open_, close) - 1e-9).all().all()),
        "low  <= min(open, close)": bool((low <= np.minimum(open_, close) + 1e-9).all().all()),
        "high >= low": bool((high >= low).all().all()),
        "all prices > 0": bool((close > 0).all().all() and (open_ > 0).all().all()),
        "bid < ask": bool((bid < ask).all().all()),
        "volume > 0": bool((volume > 0).all().all()),
        "liquidity_score in [0.1, 1.0]": bool(p("liquidity_score").stack().between(0.1, 1.0).all()),
        "event_intensity in [0, 3]": bool(p("event_intensity").stack().between(0, 3).all()),
        "event_flag == (event_intensity > 1)": bool(
            (p("event_flag") == (p("event_intensity") > 1).astype(float)).all().all()),
        "no duplicate (date, asset)": not frame.duplicated(["date", "asset_id"]).any(),
        "one row per asset per date": bool((frame.groupby("date").size() == len(p.assets)).all()),
        "no nulls": int(frame.isna().sum().sum()) == 0,
        # The engine's validate_market rejects a sector change, and sector-neutral
        # construction assumes a fixed membership, so both are asserted here.
        "each asset keeps one sector": bool((frame.groupby("asset_id").sector.nunique() == 1).all()),
        "asset universe constant over time": bool(
            frame.groupby("date").asset_id.apply(lambda s: tuple(sorted(s))).nunique() == 1),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # ---- recompute every derived field using only data through t --------------
    sector_of = p.sector
    mkt = ret1.mean(axis=1)
    sector_ret = ret1.T.groupby(sector_of).transform("mean").T

    recon = {
        "return_1d": ret1,
        "return_5d": close / close.shift(5) - 1.0,
        "return_20d": close / close.shift(20) - 1.0,
        "price_strength_20": close / close.rolling(20).mean() - 1.0,
        "price_strength_60": close / close.rolling(60).mean() - 1.0,
        "realized_vol_5d": ret1.rolling(5).std(ddof=1).clip(lower=0.001),
        "realized_vol_20d": ret1.rolling(20).std(ddof=1).clip(lower=0.001),
        "volume_zscore": (volume - volume.rolling(20).mean()) / volume.rolling(20).std(ddof=0),
        "volume_trend": volume.rolling(5).mean() / volume.rolling(20).mean() - 1.0,
        "intraday_range": (high - low) / close,
        "spread_bps": (ask - bid) / close * 10_000.0,
        "market_return": pd.DataFrame({a: mkt for a in p.assets}),
        "sector_return": sector_ret,
        "market_volatility": pd.DataFrame(
            {a: mkt.rolling(20).std(ddof=1).clip(lower=0.001) for a in p.assets}),
        "sector_dispersion": ret1.T.groupby(sector_of).transform("std", ddof=0).T,
    }

    print(f"\n===== FIELD DEFINITION RECONSTRUCTION ({split}) =====")
    print("  Recomputed from raw data using only observations up to and including t.")
    print(f"  {'field':20s} {'max|abs err|':>13s} {'corr':>9s} {'n compared':>11s}  verdict")
    rows = []
    for field, rebuilt in recon.items():
        supplied = p(field)
        a, b = supplied.align(rebuilt, join="inner")
        mask = np.isfinite(a.to_numpy(float)) & np.isfinite(b.to_numpy(float))
        av, bv = a.to_numpy(float)[mask], b.to_numpy(float)[mask]
        # skip the rolling warm-up window, which used unsupplied earlier history
        err = float(np.max(np.abs(av - bv))) if av.size else np.nan
        corr = float(np.corrcoef(av, bv)[0, 1]) if av.size > 2 and np.std(bv) > 0 else np.nan
        verdict = ("exact" if err < 1e-8 else "matches (fp)" if err < 1e-6
                   else "close" if corr > 0.999 else "MISMATCH")
        print(f"  {field:20s} {err:13.2e} {corr:9.6f} {av.size:11d}  {verdict}")
        rows.append(dict(field=field, max_abs_error=err, correlation=corr,
                         n_compared=int(av.size), verdict=verdict))

    # ---- direct look-ahead test ----------------------------------------------
    # For each field, compare how well it is explained by data through t versus
    # data through t+1. A field built with tomorrow's observation would correlate
    # markedly better with the forward-shifted reconstruction.
    print(f"\n===== LOOK-AHEAD TEST ({split}) =====")
    print("  corr(supplied, rebuilt using data <= t) vs corr(supplied, rebuilt <= t+1)")
    print(f"  {'field':20s} {'corr(<=t)':>11s} {'corr(<=t+1)':>12s}  verdict")
    leak_rows = []
    for field in ["return_1d", "return_5d", "price_strength_20", "realized_vol_20d",
                  "volume_zscore", "market_return", "market_volatility"]:
        supplied = p(field)
        past, future = recon[field], recon[field].shift(-1)
        out = {}
        for label, cand in (("past", past), ("future", future)):
            a, b = supplied.align(cand, join="inner")
            m = np.isfinite(a.to_numpy(float)) & np.isfinite(b.to_numpy(float))
            av, bv = a.to_numpy(float)[m], b.to_numpy(float)[m]
            out[label] = float(np.corrcoef(av, bv)[0, 1]) if av.size > 2 and np.std(bv) > 0 else np.nan
        verdict = "clean" if out["past"] >= out["future"] - 1e-6 else "LEAK SUSPECTED"
        print(f"  {field:20s} {out['past']:11.6f} {out['future']:12.6f}  {verdict}")
        leak_rows.append(dict(field=field, corr_past=out["past"],
                              corr_future=out["future"], verdict=verdict))

    OUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"integrity_reconstruction_{split}.csv", index=False)
    pd.DataFrame(leak_rows).to_csv(OUT / f"integrity_lookahead_{split}.csv", index=False)
    (OUT / f"integrity_invariants_{split}.json").write_text(json.dumps(checks, indent=2))
    return result


if __name__ == "__main__":
    for split in ("development", "validation"):
        audit(split)
        print()
