"""Options strategy backtester package."""

from .backtest import BacktestConfig, BacktestEngine
from .data import MarketData, VolSurface
from .instruments import (
    ButterflyStrategy,
    CalendarSpreadStrategy,
    OptionContract,
    OptionLeg,
    OptionStrategy,
    Portfolio,
    RatioSpreadStrategy,
    VerticalSpreadStrategy,
)
from .metrics import PerformanceMetrics, PnLAttributionEngine, BreakevenAnalyzer
from .models import BlackScholesModel, BachelierModel, SurfaceBumpModel
from .visualize import VisualizationEngine

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "MarketData",
    "VolSurface",
    "OptionContract",
    "OptionLeg",
    "OptionStrategy",
    "CalendarSpreadStrategy",
    "VerticalSpreadStrategy",
    "RatioSpreadStrategy",
    "ButterflyStrategy",
    "Portfolio",
    "PerformanceMetrics",
    "PnLAttributionEngine",
    "BreakevenAnalyzer",
    "BlackScholesModel",
    "BachelierModel",
    "SurfaceBumpModel",
    "VisualizationEngine",
]
