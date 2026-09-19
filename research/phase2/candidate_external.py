"""Disrupt Quant 2026 candidate strategy.

Architecture
------------
1. Strict-sector-neutral carry core using within-sector carry ranks.
2. Carry signal refreshed every five observed sessions.
3. Daily one-day reversal overlay, gated by an expanding historical estimate
   of reversal alpha relative to an estimated round-trip execution hurdle.

The expanding regression is maintained with sufficient statistics so each
new decision is inexpensive. On the first call, state is initialized entirely
from the supplied history; no warm-up calls or external files are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL_NAV_REFERENCE = 1_000_000.0
IMPACT_COEFFICIENT = 0.10
CARRY_REFRESH = 5
CARRY_GROSS = 1.0
REVERSAL_SIDE_WEIGHT = 0.04
COST_SAFETY_MARGIN = 1.5
MIN_CALIBRATION_SESSIONS = 126

# Per-period module state. The evaluator resets the module between periods.
_STATE = {
    "initialized": False,
    "last_current_date": None,
    "sxy": 0.0,
    "sxx": 0.0,
    "sessions": 0,
}


def _sector_demeaned_reversal_z(latest: pd.DataFrame) -> pd.Series:
    frame = latest.set_index("asset_id")
    signal = -frame["return_1d"].astype(float)
    sector_demeaned = signal - signal.groupby(frame["sector"]).transform("mean")
    sigma = float(sector_demeaned.std(ddof=0))

    if not np.isfinite(sigma) or sigma <= 0:
        return pd.Series(np.nan, index=frame.index)

    return (sector_demeaned - sector_demeaned.mean()) / sigma


def _strict_sector_rank_weights(latest: pd.DataFrame) -> pd.Series:
    """Exactly sector-neutral carry weights with gross exposure 1.0."""
    assets = sorted(latest["asset_id"].unique())
    frame = latest.set_index("asset_id")
    signal = frame["carry_score"].astype(float)
    sector = frame["sector"]

    parts = []
    for sector_name in sector.unique():
        ids = sector[sector == sector_name].index
        ranks = signal.loc[ids].rank(method="average")
        parts.append(ranks - ranks.mean())

    weights = pd.concat(parts).reindex(assets).fillna(0.0)
    denominator = float(weights.abs().sum())

    if denominator <= 0:
        return weights * 0.0

    return weights / denominator * CARRY_GROSS


def _reset_and_initialize(history: pd.DataFrame, current_date: pd.Timestamp) -> None:
    """Initialize expanding-regression sufficient statistics from history."""
    h = history.sort_values(["date", "asset_id"]).copy()
    dates = pd.DatetimeIndex(h["date"].unique())

    sxy = 0.0
    sxx = 0.0
    sessions = 0

    # Every pair ending no later than current_date is observable now.
    for i in range(len(dates) - 1):
        predictor = h[h["date"] == dates[i]]
        target = h[h["date"] == dates[i + 1]].set_index("asset_id")

        z = _sector_demeaned_reversal_z(predictor)
        r = target["close"].astype(float) / target["open"].astype(float) - 1.0
        r = r.reindex(z.index)

        finite = np.isfinite(z.to_numpy()) & np.isfinite(r.to_numpy())
        if not finite.any():
            continue

        x = z.to_numpy(float)[finite]
        y = r.to_numpy(float)[finite]
        sxy += float(np.dot(x, y))
        sxx += float(np.dot(x, x))
        sessions += 1

    _STATE.update({
        "initialized": True,
        "last_current_date": pd.Timestamp(current_date),
        "sxy": sxy,
        "sxx": sxx,
        "sessions": sessions,
    })


def _update_state(history: pd.DataFrame, current_date: pd.Timestamp) -> None:
    """Add the one newly completed predictor/target pair since the last call."""
    current_date = pd.Timestamp(current_date)

    # Robustly reinitialize if a fresh/non-monotonic period is run in the same process.
    if (
        not _STATE["initialized"]
        or _STATE["last_current_date"] is None
        or current_date <= _STATE["last_current_date"]
    ):
        _reset_and_initialize(history, current_date)
        return

    previous_date = pd.Timestamp(_STATE["last_current_date"])
    predictor = history[history["date"] == previous_date]
    target = history[history["date"] == current_date].set_index("asset_id")

    if predictor.empty or target.empty:
        _reset_and_initialize(history, current_date)
        return

    z = _sector_demeaned_reversal_z(predictor)
    r = target["close"].astype(float) / target["open"].astype(float) - 1.0
    r = r.reindex(z.index)

    finite = np.isfinite(z.to_numpy()) & np.isfinite(r.to_numpy())
    if finite.any():
        x = z.to_numpy(float)[finite]
        y = r.to_numpy(float)[finite]
        _STATE["sxy"] += float(np.dot(x, y))
        _STATE["sxx"] += float(np.dot(x, x))
        _STATE["sessions"] += 1

    _STATE["last_current_date"] = current_date


def _current_reversal_slope(history: pd.DataFrame, current_date: pd.Timestamp) -> float | None:
    _update_state(history, current_date)

    if _STATE["sessions"] < MIN_CALIBRATION_SESSIONS or _STATE["sxx"] <= 0:
        return None

    return float(_STATE["sxy"] / _STATE["sxx"])


def _estimated_one_way_rate(row: pd.Series, weight_abs: float) -> float:
    """Approximate supplied one-way cost using only current-close observables."""
    participation = (
        abs(weight_abs)
        * INITIAL_NAV_REFERENCE
        / (float(row["volume"]) * float(row["close"]))
    )

    impact = (
        IMPACT_COEFFICIENT
        * float(row["realized_vol_20d"])
        * np.sqrt(participation)
        / np.sqrt(np.clip(float(row["liquidity_score"]), 0.1, 1.0))
    )

    return float(row["spread_bps"]) / 20_000.0 + impact


def _reversal_overlay(latest: pd.DataFrame, slope: float) -> pd.Series:
    assets = sorted(latest["asset_id"].unique())
    frame = latest.set_index("asset_id").copy()
    z = _sector_demeaned_reversal_z(latest)
    weights = pd.Series(0.0, index=assets)

    if not np.isfinite(z.to_numpy()).any():
        return weights

    for sector_name in frame["sector"].unique():
        ids = frame.index[frame["sector"] == sector_name]
        sector_z = z.loc[ids]

        long_asset = sector_z.idxmax()
        short_asset = sector_z.idxmin()

        predicted_spread = slope * (
            float(sector_z.loc[long_asset]) - float(sector_z.loc[short_asset])
        )

        long_cost = _estimated_one_way_rate(frame.loc[long_asset], REVERSAL_SIDE_WEIGHT)
        short_cost = _estimated_one_way_rate(frame.loc[short_asset], REVERSAL_SIDE_WEIGHT)
        round_trip_hurdle = 2.0 * (long_cost + short_cost)

        if predicted_spread > COST_SAFETY_MARGIN * round_trip_hurdle:
            weights.loc[long_asset] += REVERSAL_SIDE_WEIGHT
            weights.loc[short_asset] -= REVERSAL_SIDE_WEIGHT

    return weights


def generate_positions(history, current_date):
    """Return target weights after the close of current_date."""
    dates = pd.DatetimeIndex(history["date"].unique())
    current_index = len(dates) - 1

    # Refresh the slow carry signal every five observed sessions. Between
    # refreshes, the same target is requested; the engine may still make small
    # trades to correct weight drift.
    carry_index = current_index - current_index % CARRY_REFRESH
    carry_date = dates[carry_index]
    carry_latest = history[history["date"] == carry_date]
    carry = _strict_sector_rank_weights(carry_latest)

    slope = _current_reversal_slope(history, pd.Timestamp(current_date))
    if slope is None or not np.isfinite(slope) or slope <= 0:
        return carry.to_dict()

    current = history[history["date"] == current_date]
    reversal = _reversal_overlay(current, slope)

    return (carry + reversal).to_dict()
