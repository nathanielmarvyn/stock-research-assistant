"""Technical indicators as pure functions over pandas Series.

No I/O here: everything takes price/volume Series and returns numbers, so it
is fully unit-testable. Returns are fractions (0.05 == 5%).
"""

from __future__ import annotations

import math

import pandas as pd

RSI_PERIOD = 14
VOLUME_LOOKBACK = 30


def _clean(value: float) -> float | None:
    """Convert NaN/inf to None so callers never see non-finite numbers."""
    return float(value) if value is not None and math.isfinite(value) else None


def price_on_or_before(close: pd.Series, when: pd.Timestamp) -> float | None:
    """Closing price on ``when`` or the last trading day before it (None if no data that early)."""
    earlier = close.loc[:when]
    return _clean(earlier.iloc[-1]) if not earlier.empty else None


def period_return(close: pd.Series, start: pd.Timestamp) -> float | None:
    """Return from the close on/before ``start`` to the latest close."""
    if close.empty:
        return None
    base = price_on_or_before(close, start)
    if not base:
        return None
    return _clean(close.iloc[-1] / base - 1)


def lookback_starts(last_date: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Start dates for the standard performance windows, measured back from ``last_date``.

    YTD is measured from the last close of the prior year.
    """
    year_end = pd.Timestamp(year=last_date.year - 1, month=12, day=31, tz=last_date.tz)
    return {
        "1M": last_date - pd.DateOffset(months=1),
        "6M": last_date - pd.DateOffset(months=6),
        "YTD": year_end,
        "1Y": last_date - pd.DateOffset(years=1),
    }


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average; NaN until ``window`` observations exist."""
    return close.rolling(window=window, min_periods=window).mean()


def latest_sma(close: pd.Series, window: int) -> float | None:
    """Most recent SMA value, or None when history is too short."""
    if len(close) < window:
        return None
    return _clean(sma(close, window).iloc[-1])


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Seeds the average gain/loss with a simple mean of the first ``period``
    changes, then applies Wilder's recursion: avg = (prev * (n - 1) + current) / n.
    Values before the seed are NaN. A window with no losses reads 100; with no
    movement at all, 50.
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    result = pd.Series(float("nan"), index=close.index)
    if len(close) <= period:
        return result

    avg_gain = gains.iloc[1 : period + 1].mean()
    avg_loss = losses.iloc[1 : period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        if avg_loss == 0:
            result.iloc[i] = 50.0 if avg_gain == 0 else 100.0
        else:
            result.iloc[i] = 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def latest_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float | None:
    """Most recent RSI value, or None when history is too short."""
    return _clean(rsi(close, period).iloc[-1]) if len(close) > period else None


def relative_volume(volume: pd.Series, lookback: int = VOLUME_LOOKBACK) -> tuple[float | None, float | None]:
    """Latest volume vs. the average of the ``lookback`` sessions before it.

    Returns ``(average_volume, ratio)``; ratio 1.5 means 50% above average.
    The latest session is excluded from its own baseline.
    """
    if len(volume) < lookback + 1:
        return None, None
    baseline = _clean(volume.iloc[-(lookback + 1) : -1].mean())
    if not baseline:
        return baseline, None
    return baseline, _clean(volume.iloc[-1] / baseline)
