"""Pricing model abstractions and implementations."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import MarketData


class PricingModel(abc.ABC):
    def __init__(self, market_data: MarketData):
        self.market_data = market_data

    @abc.abstractmethod
    def price(self, option, date: datetime) -> float:
        ...

    @abc.abstractmethod
    def greeks(self, option, date: datetime) -> Dict[str, float]:
        ...


@dataclass
class BlackScholesModel(PricingModel):
    market_data: MarketData

    def _inputs(self, option, date: datetime) -> Tuple[float, float, float, float, float]:
        S = self.market_data.get_spot(date)
        r = self.market_data.get_rate(date)
        q = self.market_data.get_dividend_yield(date)
        vol = self.market_data.get_vol_surface(date).iv(option.strike, option.expiry)
        T = max(1e-8, (pd.Timestamp(option.expiry) - pd.Timestamp(date)).days / 365.0)
        return S, r, q, vol, T

    def price(self, option, date: datetime) -> float:
        S, r, q, vol, T = self._inputs(option, date)
        if T <= 0:
            intrinsic = max(0.0, (S - option.strike) if option.option_type == "call" else (option.strike - S))
            return intrinsic * option.contract_size
        d1 = (np.log(S / option.strike) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        if option.option_type == "call":
            price = np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * option.strike * norm.cdf(d2)
        else:
            price = np.exp(-r * T) * option.strike * norm.cdf(-d2) - np.exp(-q * T) * S * norm.cdf(-d1)
        return price * option.contract_size

    def greeks(self, option, date: datetime) -> Dict[str, float]:
        S, r, q, vol, T = self._inputs(option, date)
        if T <= 0:
            return {g: 0.0 for g in ["delta", "gamma", "vega", "theta", "rho"]}
        d1 = (np.log(S / option.strike) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        pdf = norm.pdf(d1)
        sign = 1 if option.option_type == "call" else -1

        delta = sign * np.exp(-q * T) * norm.cdf(sign * d1)
        gamma = np.exp(-q * T) * pdf / (S * vol * np.sqrt(T))
        vega = S * np.exp(-q * T) * pdf * np.sqrt(T) * 0.01
        theta = (
            -S * pdf * vol * np.exp(-q * T) / (2 * np.sqrt(T))
            - sign * (r * option.strike * np.exp(-r * T) * norm.cdf(sign * d2))
            + sign * (q * S * np.exp(-q * T) * norm.cdf(sign * d1))
        ) / 365.0
        rho = sign * option.strike * T * np.exp(-r * T) * norm.cdf(sign * d2) * 0.01

        scale = option.contract_size
        return {
            "delta": delta * scale,
            "gamma": gamma * scale,
            "vega": vega * scale,
            "theta": theta * scale,
            "rho": rho * scale,
        }


@dataclass
class BachelierModel(PricingModel):
    market_data: MarketData

    def _inputs(self, option, date: datetime) -> Tuple[float, float, float, float, float]:
        S = self.market_data.get_spot(date)
        r = self.market_data.get_rate(date)
        q = self.market_data.get_dividend_yield(date)
        vol = self.market_data.get_vol_surface(date).iv(option.strike, option.expiry)
        T = max(1e-8, (pd.Timestamp(option.expiry) - pd.Timestamp(date)).days / 365.0)
        return S, r, q, vol, T

    def price(self, option, date: datetime) -> float:
        S, r, q, vol, T = self._inputs(option, date)
        forward = S * np.exp((r - q) * T)
        std = vol * np.sqrt(T)
        d = (forward - option.strike) / std
        sign = 1 if option.option_type == "call" else -1
        price = np.exp(-r * T) * (sign * std * norm.pdf(d) + (forward - option.strike) * norm.cdf(sign * d))
        return price * option.contract_size

    def greeks(self, option, date: datetime) -> Dict[str, float]:
        S, r, q, vol, T = self._inputs(option, date)
        forward = S * np.exp((r - q) * T)
        std = vol * np.sqrt(T)
        d = (forward - option.strike) / std
        sign = 1 if option.option_type == "call" else -1
        pdf = norm.pdf(d)

        delta = np.exp(-r * T) * np.exp((r - q) * T) * norm.cdf(sign * d)
        gamma = np.exp(-r * T) * np.exp((r - q) * T) * pdf / std
        vega = np.exp(-r * T) * pdf * np.sqrt(T) * 0.01
        theta = -r * self.price(option, date) / 365.0
        rho = -T * self.price(option, date) * 0.01

        scale = option.contract_size
        return {
            "delta": delta * scale,
            "gamma": gamma * scale,
            "vega": vega * scale,
            "theta": theta * scale,
            "rho": rho * scale,
        }


class SurfaceBumpModel(PricingModel):
    """Model-free bump-and-revalue Greeks using a base model and surface bumps."""

    def __init__(self, market_data: MarketData, base_model: Optional[PricingModel] = None, bump_size: float = 0.01):
        super().__init__(market_data)
        self.base_model = base_model or BlackScholesModel(market_data)
        self.bump_size = bump_size

    def price(self, option, date: datetime) -> float:
        return self.base_model.price(option, date)

    def greeks(self, option, date: datetime) -> Dict[str, float]:
        spot = self.market_data.get_spot(date)
        base_price = self.base_model.price(option, date)

        bump = spot * self.bump_size
        up_price = self.base_model.price(option, date)
        # temporarily shift spot in market data by creating a shadow copy
        temp_data = MarketData(
            spot_prices=self.market_data.spot_prices.copy(),
            risk_free_rates=self.market_data.risk_free_rates,
            dividend_yields=self.market_data.dividend_yields,
            vol_surfaces=self.market_data.vol_surfaces,
        )
        temp_data.spot_prices.loc[pd.Timestamp(date)] = spot + bump
        bumped_model = BlackScholesModel(temp_data)
        up_price = bumped_model.price(option, date)
        down_price = base_price if bump == 0 else None
        temp_data.spot_prices.loc[pd.Timestamp(date)] = spot - bump
        down_price = bumped_model.price(option, date)

        delta = (up_price - down_price) / (2 * bump)
        gamma = (up_price - 2 * base_price + down_price) / (bump ** 2)

        vol_bump = self.bump_size
        vol_surface = self.market_data.get_vol_surface(date)
        original_vol = vol_surface.iv(option.strike, option.expiry)
        bumped_vol_surface = MarketData(
            spot_prices=self.market_data.spot_prices,
            risk_free_rates=self.market_data.risk_free_rates,
            dividend_yields=self.market_data.dividend_yields,
            vol_surfaces={k: v for k, v in self.market_data.vol_surfaces.items()},
        ).get_vol_surface(date)
        # approximate bump by shifting grid
        bumped_vol_surface.vols = bumped_vol_surface.vols + vol_bump
        bumped_model_vol = BlackScholesModel(
            MarketData(
                self.market_data.spot_prices,
                self.market_data.risk_free_rates,
                self.market_data.dividend_yields,
                {date: bumped_vol_surface},
            )
        )
        bumped_price = bumped_model_vol.price(option, date)
        vega = (bumped_price - base_price) / vol_bump * 0.01

        theta = (self.base_model.price(option, pd.Timestamp(date) + pd.Timedelta(days=1)) - base_price) / 1 * -1
        rho = 0.0

        scale = option.contract_size
        return {
            "delta": delta * scale,
            "gamma": gamma * scale,
            "vega": vega * scale,
            "theta": theta * scale,
            "rho": rho,
        }


__all__ = ["PricingModel", "BlackScholesModel", "BachelierModel", "SurfaceBumpModel"]
