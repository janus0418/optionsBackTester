import pandas as pd
import numpy as np

from backtester.data import MarketData, VolSurface
from backtester.instruments import OptionContract
from backtester.models import BlackScholesModel, BachelierModel


def _sample_market():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    spot = pd.Series([100, 101, 102], index=dates)
    rates = pd.Series(0.01, index=dates)
    vols = np.array([[0.2, 0.21, 0.22], [0.19, 0.2, 0.21], [0.18, 0.19, 0.2]])
    expiries = [dates[-1] + pd.Timedelta(days=30), dates[-1] + pd.Timedelta(days=60), dates[-1] + pd.Timedelta(days=90)]
    strikes = [90, 100, 110]
    surfaces = {dates[0]: VolSurface(dates[0], expiries, strikes, vols)}
    md = MarketData(spot, rates, vol_surfaces=surfaces)
    return md, dates[0]


def test_black_scholes_price_positive():
    md, date = _sample_market()
    model = BlackScholesModel(md)
    option = OptionContract("SPY", "call", "european", 100, md.time_index[-1] + pd.Timedelta(days=30))
    price = model.price(option, date)
    assert price > 0


def test_black_scholes_put_call_parity_approx():
    md, date = _sample_market()
    model = BlackScholesModel(md)
    expiry = md.time_index[-1] + pd.Timedelta(days=30)
    call = OptionContract("SPY", "call", "european", 100, expiry)
    put = OptionContract("SPY", "put", "european", 100, expiry)
    call_price = model.price(call, date)
    put_price = model.price(put, date)
    spot = md.get_spot(date)
    rate = md.get_rate(date)
    parity_lhs = call_price - put_price
    parity_rhs = spot - 100 * np.exp(-rate * ((expiry - date).days / 365))
    assert abs(parity_lhs - parity_rhs) < 1.0


def test_bachelier_price_monotonic():
    md, date = _sample_market()
    model = BachelierModel(md)
    expiry = md.time_index[-1] + pd.Timedelta(days=30)
    option_itm = OptionContract("SPY", "call", "european", 90, expiry)
    option_otm = OptionContract("SPY", "call", "european", 110, expiry)
    assert model.price(option_itm, date) > model.price(option_otm, date)
