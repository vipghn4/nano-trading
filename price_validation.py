"""Shared validation rules for market prices."""

import numpy as np


def valid_price_mask(values: np.ndarray) -> np.ndarray:
    """Return true for finite, strictly positive prices."""
    values = np.asarray(values)
    return np.isfinite(values) & (values > 0.0)


def is_valid_price(value: float) -> bool:
    """Return whether one price is finite and strictly positive."""
    return bool(valid_price_mask(value))


def validate_active_prices(prices: np.ndarray, holdings: np.ndarray) -> None:
    """Require valid prices for every non-zero holding."""
    active_prices = prices[holdings != 0.0]
    if not valid_price_mask(active_prices).all():
        raise ValueError("Active holdings require finite, strictly positive market prices")
