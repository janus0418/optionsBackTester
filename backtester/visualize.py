"""Visualization utilities for PnL and risk profiles."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class VisualizationEngine:
    def plot_pnl(self, pnl_series: pd.Series) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pnl_series.index, y=pnl_series.values, mode="lines", name="PnL"))
        fig.update_layout(title="Cumulative PnL", xaxis_title="Date", yaxis_title="PnL")
        return fig

    def plot_drawdown(self, returns: pd.Series) -> go.Figure:
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode="lines", name="Drawdown"))
        fig.update_layout(title="Drawdown", yaxis_title="Drawdown")
        return fig

    def plot_pnl_attribution(self, pnl_attr: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        for col in [c for c in pnl_attr.columns if c != "total"]:
            fig.add_trace(go.Bar(x=pnl_attr.index, y=pnl_attr[col], name=col))
        fig.update_layout(barmode="stack", title="PnL Attribution")
        return fig

    def risk_profile(self, strategy, date: datetime, model, pct: float = 0.3, steps: int = 50) -> go.Figure:
        spot = model.market_data.get_spot(date)
        prices = np.linspace(spot * (1 - pct), spot * (1 + pct), steps)
        values = []
        for p in prices:
            model.market_data.spot_prices.loc[pd.Timestamp(date)] = p
            values.append(strategy.value(date, model))
        model.market_data.spot_prices.loc[pd.Timestamp(date)] = spot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=np.array(values) - values[0], mode="lines", name="P&L"))
        fig.update_layout(title="Risk Profile", xaxis_title="Spot", yaxis_title="PnL vs current")
        return fig


__all__ = ["VisualizationEngine"]
