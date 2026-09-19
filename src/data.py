"""Data loading, panel reshaping, forward-return decomposition and the cost model.

Every function here is pure and uses only information available at or before the
stated observation date. Nothing in this module looks ahead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Engine constants, mirrored from starter/backtester.py so research and the
# strategy price trades with exactly the same arithmetic as the evaluator.
IMPACT_COEFFICIENT = 0.10
SPREAD_DIVISOR = 20_000.0
LIQUIDITY_FLOOR, LIQUIDITY_CAP = 0.1, 1.0

MAX_ASSET_WEIGHT = 0.20
MAX_GROSS = 2.00
MAX_ABS_NET = 1.00
MAX_ABS_SECTOR_NET = 0.50

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(split: str = "development", root: Path | None = None) -> pd.DataFrame:
    """Load one split, sorted exactly as the engine sorts it."""
    root = Path(root) if root is not None else REPO_ROOT
    frame = pd.read_csv(root / "data" / f"{split}.csv", parse_dates=["date"])
    return frame.sort_values(["date", "asset_id"]).reset_index(drop=True)


def load_combined(root: Path | None = None) -> pd.DataFrame:
    """Development followed by validation, as the engine concatenates them."""
    dev = load_split("development", root)
    val = load_split("validation", root)
    return pd.concat([dev, val], ignore_index=True)


class Panel:
    """Long market data reshaped to date x asset matrices.

    Columns are held in the engine's sorted asset order so that any numpy array
    derived from this class lines up with ``sorted(data.asset_id.unique())``.
    """

    def __init__(self, frame: pd.DataFrame):
        frame = frame.sort_values(["date", "asset_id"])
        self._frame = frame
        self.assets = sorted(frame.asset_id.unique())
        self.sector = frame.groupby("asset_id").sector.first().reindex(self.assets)
        self.dates = pd.DatetimeIndex(frame.date.unique())
        self._cache: dict[str, pd.DataFrame] = {}

    def __call__(self, field: str) -> pd.DataFrame:
        if field not in self._cache:
            wide = self._frame.pivot(index="date", columns="asset_id", values=field)
            self._cache[field] = wide.reindex(columns=self.assets)
        return self._cache[field]

    # ---- return decomposition -------------------------------------------------
    # The engine gives a NEWLY requested position only the open(t+1) -> close(t+1)
    # move. The close(t) -> open(t+1) gap accrues solely to positions already
    # held. Signal research must therefore be scored against `forward_open_close`,
    # not against close-to-close returns.

    def forward_close_close(self, k: int = 1) -> pd.DataFrame:
        """close(t) -> close(t+k). Not fully earnable by a new position."""
        close = self("close")
        return close.shift(-k) / close - 1.0

    def forward_open_close(self, k: int = 1) -> pd.DataFrame:
        """open(t+k) -> close(t+k). What a position opened at t+k actually earns."""
        return self("close").shift(-k) / self("open").shift(-k) - 1.0

    def forward_gap(self, k: int = 1) -> pd.DataFrame:
        """close(t+k-1) -> open(t+k). Earned only by pre-existing holdings."""
        return self("open").shift(-k) / self("close").shift(-(k - 1) or None) - 1.0

    def overnight_gap(self) -> pd.DataFrame:
        """close(t) -> open(t+1)."""
        return self("open").shift(-1) / self("close") - 1.0

    # ---- neutralisation -------------------------------------------------------
    def demean_cross_section(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Remove the equal-weight cross-sectional mean (market-factor hedge)."""
        return wide.sub(wide.mean(axis=1), axis=0)

    def demean_sector(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Remove each sector's cross-sectional mean on each date.

        Fields that are constant within a sector (for example ``sector_return``
        or ``sector_dispersion``) become identically zero. Callers must treat an
        all-zero result as 'not defined under sector neutralisation' rather than
        as a signal; ``is_degenerate`` below makes that check explicit.
        """
        groups = self.sector.reindex(wide.columns)
        return wide.sub(wide.T.groupby(groups).transform("mean").T)

    @staticmethod
    def is_degenerate(wide: pd.DataFrame, tol: float = 1e-12) -> bool:
        """True when a field carries no usable cross-sectional variation."""
        spread = wide.std(axis=1, ddof=0)
        return bool(np.nanmax(spread.to_numpy()) < tol) if len(spread) else True


def one_way_cost_rate(observed: pd.DataFrame, delta: np.ndarray, nav: float,
                      multiplier: float = 1.0) -> np.ndarray:
    """One-way cost rate per unit traded, replicating ``starter.backtester.trading_cost``.

    Parameters
    ----------
    observed : the close row of the DECISION date t (the engine passes
        ``data.iloc[(i-1)*n : i*n]``, i.e. the last observed close, not t+1).
    delta : signed traded notional divided by pre-trade opening NAV.
    nav : pre-trade opening NAV in dollars.

    Returns the rate, so total cost fraction is ``abs(delta) * rate``.
    """
    volume = observed.volume.to_numpy(float)
    close = observed.close.to_numpy(float)
    participation = np.abs(delta) * nav / (volume * close)
    impact = IMPACT_COEFFICIENT * observed.realized_vol_20d.to_numpy(float) * np.sqrt(participation)
    impact /= np.sqrt(np.clip(observed.liquidity_score.to_numpy(float), LIQUIDITY_FLOOR, LIQUIDITY_CAP))
    return multiplier * (observed.spread_bps.to_numpy(float) / SPREAD_DIVISOR + impact)


def cost_fraction(observed: pd.DataFrame, delta: np.ndarray, nav: float,
                  multiplier: float = 1.0) -> np.ndarray:
    """Per-asset cost as a fraction of pre-trade opening NAV."""
    return np.abs(delta) * one_way_cost_rate(observed, delta, nav, multiplier)
