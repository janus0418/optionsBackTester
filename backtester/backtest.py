"""Backtest engine orchestrating simulation and analytics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional

import pandas as pd

from .instruments import Portfolio
from .metrics import PerformanceMetrics, PnLAttributionEngine
from .models import PricingModel


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float = 0.0
    rolling_windows: Optional[Iterable[int]] = None


class BacktestEngine:
    def __init__(self, portfolio: Portfolio, model: PricingModel, config: BacktestConfig):
        self.portfolio = portfolio
        self.model = model
        self.config = config

    def run(self) -> Dict[str, pd.DataFrame]:
        dates = [d for d in self.model.market_data.time_index if self.config.start_date <= d <= self.config.end_date]
        portfolio_values = []
        greek_records: Dict[str, list] = {g: [] for g in ["delta", "gamma", "vega", "theta", "rho"]}
        for date in dates:
            value = self.portfolio.value(date, self.model)
            portfolio_values.append(value)
            greeks = self.portfolio.greeks(date, self.model)
            for greek in greek_records.keys():
                greek_records[greek].append(greeks.get(greek, 0.0))
        value_series = pd.Series(portfolio_values, index=dates, name="portfolio_value")
        greek_series = {k: pd.Series(v, index=dates, name=k) for k, v in greek_records.items()}

        returns = value_series.pct_change().fillna(0)
        metrics = PerformanceMetrics(returns)

        pnl_attr_engine = PnLAttributionEngine(
            spot=self.model.market_data.spot_prices.loc[dates], rates=self.model.market_data.risk_free_rates.loc[dates]
        )
        pnl_attr = pnl_attr_engine.attribute(value_series, greek_series)

        rolling = {}
        if self.config.rolling_windows:
            for w in self.config.rolling_windows:
                rolling[f"rolling_sharpe_{w}"] = metrics.rolling_sharpe(w)
                rolling[f"rolling_vol_{w}"] = metrics.rolling_vol(w)
                rolling[f"rolling_mdd_{w}"] = metrics.rolling_max_drawdown(w)

        results = {
            "values": value_series,
            "returns": returns,
            "greeks": pd.DataFrame(greek_series),
            "pnl_attribution": pd.DataFrame(
                {
                    "total": pnl_attr.total_pnl,
                    "delta": pnl_attr.delta,
                    "gamma": pnl_attr.gamma,
                    "vega": pnl_attr.vega,
                    "theta": pnl_attr.theta,
                    "rho": pnl_attr.rho,
                    "residual": pnl_attr.residual,
                }
            ),
            "metrics": pd.Series(
                {
                    "cumulative_pnl": metrics.cumulative_pnl(),
                    "annualized_return": metrics.annualized_return(),
                    "annualized_vol": metrics.annualized_vol(),
                    "sharpe_ratio": metrics.sharpe_ratio(),
                    "max_drawdown": metrics.max_drawdown(),
                }
            ),
            "rolling": pd.DataFrame(rolling),
        }
        return results


__all__ = ["BacktestEngine", "BacktestConfig"]
