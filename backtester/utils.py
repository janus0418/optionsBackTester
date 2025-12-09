"""Helper utilities."""
from __future__ import annotations

from datetime import datetime

import pandas as pd


def year_fraction(start: datetime, end: datetime, basis: int = 365) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / basis


__all__ = ["year_fraction"]
