"""Market data and volatility surface abstractions."""
from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def _year_fraction(start: datetime, end: datetime) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.0


@dataclasses.dataclass
class VolSurface:
    """Implied volatility surface with simple bilinear interpolation.

    Parameters
    ----------
    reference_date:
        Valuation date for which the surface is calibrated.
    expiries:
        Iterable of expiry datetimes.
    strikes:
        Iterable of strikes used for calibration.
    vols:
        2D array-like of shape (n_expiries, n_strikes) holding implied vols.
    """

    reference_date: datetime
    expiries: Iterable[datetime]
    strikes: Iterable[float]
    vols: np.ndarray

    def __post_init__(self) -> None:
        self.expiries = pd.to_datetime(pd.Index(self.expiries)).to_pydatetime()
        self.strikes = np.asarray(self.strikes, dtype=float)
        self.vols = np.asarray(self.vols, dtype=float)
        if self.vols.shape != (len(self.expiries), len(self.strikes)):
            raise ValueError("vol grid shape mismatch vs strikes/expiries")

    def iv(self, strike: float, expiry: datetime) -> float:
        """Return implied vol via bilinear interpolation across strike and expiry."""
        target_expiry = pd.Timestamp(expiry)
        expiry_times = np.array([_year_fraction(self.reference_date, e) for e in self.expiries])
        target_time = _year_fraction(self.reference_date, target_expiry)

        if target_time <= expiry_times.min():
            lower_idx = upper_idx = expiry_times.argmin()
            weight = 0.0
        elif target_time >= expiry_times.max():
            lower_idx = upper_idx = expiry_times.argmax()
            weight = 0.0
        else:
            upper_idx = np.searchsorted(expiry_times, target_time)
            lower_idx = upper_idx - 1
            t0, t1 = expiry_times[lower_idx], expiry_times[upper_idx]
            weight = (target_time - t0) / (t1 - t0)

        def _strike_interp(row: np.ndarray) -> float:
            interp = interp1d(self.strikes, row, fill_value="extrapolate")
            return float(interp(strike))

        lower_vol = _strike_interp(self.vols[lower_idx])
        upper_vol = _strike_interp(self.vols[upper_idx])
        return (1 - weight) * lower_vol + weight * upper_vol


class MarketData:
    """Container for time series market data and vol surfaces."""

    def __init__(
        self,
        spot_prices: pd.Series,
        risk_free_rates: pd.Series,
        dividend_yields: Optional[pd.Series] = None,
        vol_surfaces: Optional[Dict[pd.Timestamp, VolSurface]] = None,
    ) -> None:
        self.spot_prices = spot_prices.sort_index()
        self.risk_free_rates = risk_free_rates.reindex(self.spot_prices.index).fillna(method="ffill")
        self.dividend_yields = (
            dividend_yields.reindex(self.spot_prices.index).fillna(method="ffill") if dividend_yields is not None else None
        )
        self.vol_surfaces = vol_surfaces or {}

    @property
    def time_index(self) -> pd.DatetimeIndex:
        return self.spot_prices.index

    def get_spot(self, date: datetime) -> float:
        return float(self.spot_prices.loc[pd.Timestamp(date)])

    def get_rate(self, date: datetime) -> float:
        return float(self.risk_free_rates.loc[pd.Timestamp(date)])

    def get_dividend_yield(self, date: datetime) -> float:
        if self.dividend_yields is None:
            return 0.0
        return float(self.dividend_yields.loc[pd.Timestamp(date)])

    def get_vol_surface(self, date: datetime) -> VolSurface:
        key = pd.Timestamp(date)
        if key not in self.vol_surfaces:
            raise KeyError(f"No vol surface for {key}")
        return self.vol_surfaces[key]

    def slice(self, start: datetime, end: datetime) -> "MarketData":
        mask = (self.time_index >= pd.Timestamp(start)) & (self.time_index <= pd.Timestamp(end))
        sliced_surfaces = {k: v for k, v in self.vol_surfaces.items() if mask[self.time_index.get_loc(k)]}
        return MarketData(
            spot_prices=self.spot_prices.loc[mask],
            risk_free_rates=self.risk_free_rates.loc[mask],
            dividend_yields=self.dividend_yields.loc[mask] if self.dividend_yields is not None else None,
            vol_surfaces=sliced_surfaces,
        )


__all__ = ["MarketData", "VolSurface"]
