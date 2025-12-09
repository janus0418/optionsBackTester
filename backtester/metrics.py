"""Performance metrics, breakevens, and PnL attribution."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    returns: pd.Series

    def cumulative_pnl(self) -> float:
        return float((1 + self.returns).prod() - 1)

    def annualized_return(self, periods_per_year: int = 252) -> float:
        cumulative = 1 + self.cumulative_pnl()
        years = len(self.returns) / periods_per_year
        return cumulative ** (1 / years) - 1 if years > 0 else 0.0

    def annualized_vol(self, periods_per_year: int = 252) -> float:
        return float(self.returns.std(ddof=0) * np.sqrt(periods_per_year))

    def sharpe_ratio(self, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
        excess = self.returns - risk_free / periods_per_year
        vol = excess.std(ddof=0)
        return float(excess.mean() / vol * np.sqrt(periods_per_year)) if vol != 0 else 0.0

    def max_drawdown(self) -> float:
        equity = (1 + self.returns).cumprod()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        return float(drawdown.min())

    def rolling_sharpe(self, window: int, periods_per_year: int = 252) -> pd.Series:
        return (
            self.returns.rolling(window).mean()
            / self.returns.rolling(window).std(ddof=0)
            * np.sqrt(periods_per_year)
        )

    def rolling_vol(self, window: int, periods_per_year: int = 252) -> pd.Series:
        return self.returns.rolling(window).std(ddof=0) * np.sqrt(periods_per_year)

    def rolling_max_drawdown(self, window: int) -> pd.Series:
        def _mdd(x: pd.Series) -> float:
            equity = (1 + x).cumprod()
            return (equity / equity.cummax() - 1).min()

        return self.returns.rolling(window).apply(_mdd, raw=False)


@dataclass
class PnLAttributionResult:
    total_pnl: pd.Series
    delta: pd.Series
    gamma: pd.Series
    vega: pd.Series
    theta: pd.Series
    rho: pd.Series
    residual: pd.Series


class PnLAttributionEngine:
    def __init__(self, spot: pd.Series, rates: pd.Series):
        self.spot = spot
        self.rates = rates

    def attribute(
        self,
        values: pd.Series,
        greeks: Dict[str, pd.Series],
        implied_vol_proxy: Optional[pd.Series] = None,
    ) -> PnLAttributionResult:
        delta_s = self.spot.diff().fillna(0)
        delta_r = self.rates.diff().fillna(0)
        delta_sigma = implied_vol_proxy.diff().fillna(0) if implied_vol_proxy is not None else pd.Series(0, index=self.spot.index)
        delta_t = 1 / 365.0

        pnl_delta = greeks.get("delta", pd.Series(0, index=self.spot.index)) * delta_s
        pnl_gamma = 0.5 * greeks.get("gamma", pd.Series(0, index=self.spot.index)) * (delta_s ** 2)
        pnl_vega = greeks.get("vega", pd.Series(0, index=self.spot.index)) * delta_sigma
        pnl_theta = greeks.get("theta", pd.Series(0, index=self.spot.index)) * delta_t
        pnl_rho = greeks.get("rho", pd.Series(0, index=self.spot.index)) * delta_r

        explained = pnl_delta + pnl_gamma + pnl_vega + pnl_theta + pnl_rho
        total_pnl = values.diff().fillna(0)
        residual = total_pnl - explained

        return PnLAttributionResult(total_pnl, pnl_delta, pnl_gamma, pnl_vega, pnl_theta, pnl_rho, residual)


class BreakevenAnalyzer:
    def __init__(self, grid_pct: float = 0.5, grid_points: int = 50):
        self.grid_pct = grid_pct
        self.grid_points = grid_points

    def find_breakevens(self, strategy, date: datetime, model, horizons: List[int]) -> Dict[int, Tuple[float, float]]:
        breakevens: Dict[int, Tuple[float, float]] = {}
        spot = model.market_data.get_spot(date)
        strikes = np.linspace(spot * (1 - self.grid_pct), spot * (1 + self.grid_pct), self.grid_points)
        current_value = strategy.value(date, model)
        for horizon in horizons:
            shifted_date = pd.Timestamp(date) + pd.Timedelta(days=horizon)
            values = []
            for s in strikes:
                temp_md = model.market_data
                temp_md.spot_prices.loc[pd.Timestamp(date)] = s
                values.append(strategy.value(date, model))
                temp_md.spot_prices.loc[pd.Timestamp(date)] = spot
            pnl_curve = np.array(values) - current_value
            sign_changes = np.where(np.diff(np.sign(pnl_curve)) != 0)[0]
            if len(sign_changes) >= 1:
                lower_idx = sign_changes[0]
                upper_idx = sign_changes[-1] if len(sign_changes) > 1 else sign_changes[0]
                breakevens[horizon] = (strikes[lower_idx], strikes[upper_idx])
            else:
                breakevens[horizon] = (np.nan, np.nan)
        return breakevens


__all__ = ["PerformanceMetrics", "PnLAttributionEngine", "BreakevenAnalyzer", "PnLAttributionResult"]
