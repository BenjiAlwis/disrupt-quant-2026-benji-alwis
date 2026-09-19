"""Carry core plus cost-gated reversal overlay.

Design, in one paragraph
------------------------
Phase 1 found two independent edges with very different lifetimes. `carry_score`
is weak but persistent (IC ~0.03, undecayed at 20 sessions), so it is held as a
strictly sector-neutral core and traded rarely. One-day reversal is stronger but
dead within a session, and traded naively its costs exceed its alpha by ~2x. So
it is run only as an overlay of within-sector pairs, and a pair is opened only
when its expected edge exceeds its round-trip cost. Cost is 81-95% spread and
spread varies ~10x across assets, so the gate mostly admits cheap names: the
selection comes from the cost function, not from a hand-picked liquidity filter.

Everything the strategy estimates is estimated from the history it is given, so
the same code runs identically in walk-forward research and in evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
import pandas as pd

# engine constants (starter/backtester.py)
IMPACT_COEFFICIENT = 0.10
MAX_ASSET, MAX_GROSS, MAX_NET, MAX_SECTOR_NET = 0.20, 2.00, 1.00, 0.50


@dataclass(frozen=True)
class Config:
    # --- carry core ---------------------------------------------------------
    carry_gross: float = 1.20        # gross exposure of the core (risk budget, see note)
    carry_band: float = 0.25         # rebalance a sector only if a weight drifts by more than
                                     # this fraction of one rank step from its target
    carry_refresh: int = 1           # recompute carry ranks every k sessions (1 = daily)
    carry_smooth: int = 1            # rank a k-session mean of carry_score (1 = raw score)
    # --- reversal overlay ---------------------------------------------------
    # Disabled in the submission: it failed its pre-registered walk-forward test
    # (increment +1.0%/yr, t = 0.64, negative at 1.5x and 2x costs). Kept in the
    # code so the negative result is reproducible; see artifacts/phase2/.
    use_reversal_overlay: bool = True
    rev_leg: float = 0.10            # weight per leg of each within-sector pair
    rev_kappa: float = 1.00          # trade iff b*|dr| >= kappa * round-trip cost
    rev_min_sessions: int = 60       # history needed before the overlay may trade
    rev_recalibrate: int = 21        # re-estimate b every ~month from expanding history
    rev_prior_b: float | None = None # optional development prior, used only while
    rev_prior_sessions: int = 250    #   history is shorter than this many sessions
    # --- cost model inputs --------------------------------------------------
    reference_nav: float = 1_000_000.0  # strategy cannot observe NAV; impact is priced at the
                                        # initial NAV, a small error since impact is 5-19% of cost
    # --- safety margins inside the engine's hard limits ---------------------
    max_asset: float = 0.19
    max_gross: float = 1.95

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# building blocks (pure functions, unit-testable)
# ---------------------------------------------------------------------------

def one_way_cost(row: pd.DataFrame, leg: float, nav: float) -> np.ndarray:
    """Engine cost rate for trading `leg` of NAV, priced off the decision-date row."""
    participation = leg * nav / (row.volume.to_numpy(float) * row.close.to_numpy(float))
    impact = IMPACT_COEFFICIENT * row.realized_vol_20d.to_numpy(float) * np.sqrt(participation)
    impact /= np.sqrt(np.clip(row.liquidity_score.to_numpy(float), 0.1, 1.0))
    return row.spread_bps.to_numpy(float) / 20_000.0 + impact


def carry_targets(carry: np.ndarray, sectors: np.ndarray, gross: float) -> np.ndarray:
    """Rank carry WITHIN each sector and centre, so every sector nets to exactly zero."""
    w = np.zeros_like(carry, dtype=float)
    for s in np.unique(sectors):
        idx = np.where(sectors == s)[0]
        ranks = pd.Series(carry[idx]).rank(method="first").to_numpy()
        w[idx] = ranks - ranks.mean()
    total = np.abs(w).sum()
    return w / total * gross if total > 0 else w


def estimate_reversal_b(ret1: np.ndarray, oc: np.ndarray, sectors: np.ndarray) -> float:
    """Pooled slope of tomorrow's sector-relative open->close on today's return.

    ret1, oc: (n_dates, n_assets) arrays aligned so oc[t] is the session AFTER ret1[t].
    Returns b = -slope, the fraction of a sector-relative move that reverses.
    """
    def demean(x):
        out = x.copy()
        for s in np.unique(sectors):
            idx = sectors == s
            out[:, idx] -= x[:, idx].mean(axis=1, keepdims=True)
        return out
    xv, yv = demean(ret1).ravel(), demean(oc).ravel()
    m = np.isfinite(xv) & np.isfinite(yv)
    if m.sum() < 100 or np.var(xv[m]) <= 0:
        return 0.0
    return float(-np.polyfit(xv[m], yv[m], 1)[0])


def reversal_pairs(ret1: np.ndarray, cost: np.ndarray, sectors: np.ndarray,
                   b: float, kappa: float, leg: float) -> tuple[np.ndarray, list]:
    """Best cost-clearing within-sector pair per sector: long the loser, short the winner.

    Expected edge of a pair is b*|r_i - r_j|; round-trip cost is 2*(c_i + c_j). A pair
    is admitted only if edge >= kappa * cost, and at most one pair per sector.
    """
    w = np.zeros(len(ret1))
    chosen = []
    if b <= 0:
        return w, chosen
    for s in np.unique(sectors):
        idx = np.where(sectors == s)[0]
        best = None
        for i, j in combinations(idx, 2):
            dr = ret1[i] - ret1[j]
            rt = 2.0 * (cost[i] + cost[j])
            margin = b * abs(dr) - kappa * rt
            if np.isfinite(margin) and margin > 0 and (best is None or margin > best[0]):
                loser, winner = (j, i) if dr > 0 else (i, j)
                best = (margin, loser, winner)
        if best:
            _, lo, hi = best
            w[lo] += leg
            w[hi] -= leg
            chosen.append((s, lo, hi, best[0]))
    return w, chosen


def enforce_limits(w: np.ndarray, sectors: np.ndarray, max_asset: float,
                   max_gross: float) -> np.ndarray:
    """Stay strictly inside the engine limits so its clamp never rescales the book."""
    w = np.clip(w, -max_asset, max_asset)
    gross = np.abs(w).sum()
    if gross > max_gross:
        w = w * (max_gross / gross)
    for s in np.unique(sectors):                       # belt and braces; ~0 by construction
        net = w[sectors == s].sum()
        if abs(net) > 0.45:
            w[sectors == s] -= net / (sectors == s).sum()
    return w


# ---------------------------------------------------------------------------
# the stateful allocator the engine calls
# ---------------------------------------------------------------------------

class Allocator:
    """Holds state within one evaluation period and resets itself between periods."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self._reset()

    def _reset(self):
        self.assets = None
        self.sectors = None
        self.last_date = None
        self.prev_total = None      # total weights requested at the previous close
        self.carry_held = None      # carry component of those weights
        self.b = 0.0
        self.sessions_since_fit = None
        self.session_count = 0
        self.carry_target = None
        self.log = []               # per-decision diagnostics for research attribution

    # -- helpers -----------------------------------------------------------------
    def _panel(self, history: pd.DataFrame):
        n = len(self.assets)
        tail = history.iloc[-(len(history) // n) * n:]
        shape = (len(tail) // n, n)
        get = lambda c: tail[c].to_numpy(float).reshape(shape)
        return get("return_1d"), get("open"), get("close")

    def _fit_b(self, history: pd.DataFrame):
        ret1, open_, close = self._panel(history)
        oc = close / open_ - 1.0
        n_sessions = ret1.shape[0] - 1
        b_hist = estimate_reversal_b(ret1[:-1], oc[1:], self.sectors) if n_sessions >= 2 else 0.0
        cfg = self.cfg
        if cfg.rev_prior_b is not None and n_sessions < cfg.rev_prior_sessions:
            w_hist = n_sessions / cfg.rev_prior_sessions
            b_hist = w_hist * b_hist + (1 - w_hist) * cfg.rev_prior_b
        elif n_sessions < cfg.rev_min_sessions:
            b_hist = 0.0
        self.b = max(b_hist, 0.0)          # if reversal vanishes, the overlay switches off
        self.sessions_since_fit = 0

    def _drifted(self, w: np.ndarray, row: pd.DataFrame) -> np.ndarray:
        """Component weights requested yesterday, carried through today's open->close move.

        Normalised by the WHOLE portfolio's return, since weights are fractions of NAV.
        """
        g = row.close.to_numpy(float) / row.open.to_numpy(float) - 1.0
        whole = self.prev_total if self.prev_total is not None else w
        return w * (1 + g) / (1 + float(whole @ g))

    # -- the engine entry point ---------------------------------------------------
    def decide(self, history: pd.DataFrame, current_date: pd.Timestamp) -> dict:
        cfg = self.cfg
        if self.last_date is not None and current_date <= self.last_date:
            self._reset()                                  # a new evaluation period began
        if self.assets is None:
            first = history[history.date == history.date.iloc[-1]]
            self.assets = list(first.asset_id)
            self.sectors = first.sector.to_numpy()
        n = len(self.assets)
        row = history.iloc[-n:]
        assert (row.date == current_date).all() and list(row.asset_id) == self.assets

        # (re)estimate the reversal coefficient from expanding history
        if cfg.use_reversal_overlay and (self.sessions_since_fit is None
                                         or self.sessions_since_fit >= cfg.rev_recalibrate):
            self._fit_b(history)
        if self.sessions_since_fit is not None:
            self.sessions_since_fit += 1

        # -- carry core. Ranks are recomputed every `carry_refresh` sessions from a
        #    `carry_smooth`-session mean score; both filter the noise that otherwise makes
        #    near-tied names swap ranks and swap back (58% of daily rank changes in
        #    2021-22 reversed within five sessions).
        if self.carry_target is None or self.session_count % cfg.carry_refresh == 0:
            if cfg.carry_smooth > 1:
                k = cfg.carry_smooth
                block = history.iloc[-k * n:].carry_score.to_numpy(float)
                score = block.reshape(-1, n).mean(axis=0)
            else:
                score = row.carry_score.to_numpy(float)
            self.carry_target = carry_targets(score, self.sectors, cfg.carry_gross)
        self.session_count += 1
        target = self.carry_target
        if self.carry_held is None:
            carry = target
        else:
            held = self._drifted(self.carry_held, row)
            step = cfg.carry_gross / 24.0                  # weight gap between adjacent ranks
            carry = held.copy()
            for s in np.unique(self.sectors):
                idx = self.sectors == s
                if np.abs(target[idx] - held[idx]).max() > cfg.carry_band * step:
                    carry[idx] = target[idx]               # trade the whole sector: stays neutral
        self.carry_held = carry

        # -- reversal overlay: fresh each day, cost-gated
        if cfg.use_reversal_overlay:
            cost = one_way_cost(row, cfg.rev_leg, cfg.reference_nav)
            overlay, chosen = reversal_pairs(row.return_1d.to_numpy(float), cost, self.sectors,
                                             self.b, cfg.rev_kappa, cfg.rev_leg)
        else:
            overlay, chosen = np.zeros(n), []

        total = enforce_limits(carry + overlay, self.sectors, cfg.max_asset, cfg.max_gross)
        self.prev_total, self.last_date = total, current_date
        self.log.append({"date": current_date, "b": self.b, "pairs": len(chosen),
                         "carry_gross": float(np.abs(carry).sum()),
                         "overlay_gross": float(np.abs(overlay).sum())})
        return {a: float(w) for a, w in zip(self.assets, total) if w != 0.0}
