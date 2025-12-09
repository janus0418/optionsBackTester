"""Option instruments and strategy abstractions."""
from __future__ import annotations

import abc
import dataclasses
from datetime import datetime
from typing import Dict, Iterable, List

from .models import PricingModel


@dataclasses.dataclass(frozen=True)
class OptionContract:
    underlying: str
    option_type: str  # "call" or "put"
    style: str
    strike: float
    expiry: datetime
    contract_size: float = 100.0


@dataclasses.dataclass
class OptionLeg:
    contract: OptionContract
    quantity: float

    def price(self, date: datetime, model: PricingModel) -> float:
        return self.quantity * model.price(self.contract, date)

    def greeks(self, date: datetime, model: PricingModel) -> Dict[str, float]:
        leg_greeks = model.greeks(self.contract, date)
        return {k: v * self.quantity for k, v in leg_greeks.items()}


class OptionStrategy(abc.ABC):
    def __init__(self, name: str, legs: Iterable[OptionLeg]) -> None:
        self.name = name
        self.legs: List[OptionLeg] = list(legs)

    def value(self, date: datetime, model: PricingModel) -> float:
        return sum(leg.price(date, model) for leg in self.legs)

    def greeks(self, date: datetime, model: PricingModel) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for leg in self.legs:
            for greek, value in leg.greeks(date, model).items():
                totals[greek] = totals.get(greek, 0.0) + value
        return totals

    @abc.abstractmethod
    def description(self) -> str:
        ...


class CalendarSpreadStrategy(OptionStrategy):
    def __init__(self, underlying: str, strike: float, near_expiry: datetime, far_expiry: datetime, quantity: float = 1.0):
        near = OptionLeg(OptionContract(underlying, "call", "european", strike, near_expiry), -quantity)
        far = OptionLeg(OptionContract(underlying, "call", "european", strike, far_expiry), quantity)
        super().__init__("calendar_spread", [near, far])

    def description(self) -> str:
        return "Long-dated call financed by short near-dated call"


class VerticalSpreadStrategy(OptionStrategy):
    def __init__(self, underlying: str, lower_strike: float, upper_strike: float, expiry: datetime, quantity: float = 1.0):
        long_leg = OptionLeg(OptionContract(underlying, "call", "european", lower_strike, expiry), quantity)
        short_leg = OptionLeg(OptionContract(underlying, "call", "european", upper_strike, expiry), -quantity)
        super().__init__("vertical_spread", [long_leg, short_leg])

    def description(self) -> str:
        return "Bull call spread"


class RatioSpreadStrategy(OptionStrategy):
    def __init__(self, underlying: str, strike_long: float, strike_short: float, expiry: datetime, ratio: float = 2.0):
        long_leg = OptionLeg(OptionContract(underlying, "call", "european", strike_long, expiry), 1.0)
        short_leg = OptionLeg(OptionContract(underlying, "call", "european", strike_short, expiry), -ratio)
        super().__init__("ratio_spread", [long_leg, short_leg])

    def description(self) -> str:
        return "Ratio spread with more shorts than longs"


class ButterflyStrategy(OptionStrategy):
    def __init__(self, underlying: str, lower: float, middle: float, upper: float, expiry: datetime, quantity: float = 1.0):
        lower_leg = OptionLeg(OptionContract(underlying, "call", "european", lower, expiry), quantity)
        middle_leg = OptionLeg(OptionContract(underlying, "call", "european", middle, expiry), -2 * quantity)
        upper_leg = OptionLeg(OptionContract(underlying, "call", "european", upper, expiry), quantity)
        super().__init__("butterfly", [lower_leg, middle_leg, upper_leg])

    def description(self) -> str:
        return "Balanced call butterfly"


class Portfolio:
    def __init__(self, strategies: Iterable[OptionStrategy], cash: float = 0.0) -> None:
        self.strategies: List[OptionStrategy] = list(strategies)
        self.cash = cash

    def value(self, date: datetime, model: PricingModel) -> float:
        return self.cash + sum(strategy.value(date, model) for strategy in self.strategies)

    def greeks(self, date: datetime, model: PricingModel) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for strat in self.strategies:
            for greek, value in strat.greeks(date, model).items():
                totals[greek] = totals.get(greek, 0.0) + value
        return totals


__all__ = [
    "OptionContract",
    "OptionLeg",
    "OptionStrategy",
    "CalendarSpreadStrategy",
    "VerticalSpreadStrategy",
    "RatioSpreadStrategy",
    "ButterflyStrategy",
    "Portfolio",
]
