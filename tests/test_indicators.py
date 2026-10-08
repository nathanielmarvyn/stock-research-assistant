"""Indicator tests against small, hand-calculated examples."""

import math

import pandas as pd
import pytest

from brief import indicators as ind


def series(values, start="2026-01-05") -> pd.Series:
    """Daily (business-day) series starting on a Monday."""
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ---------------------------------------------------------------- returns


def test_period_return_basic() -> None:
    close = series([100, 105, 110])
    assert ind.period_return(close, close.index[0]) == pytest.approx(0.10)


def test_period_return_uses_prior_trading_day_for_weekend_start() -> None:
    close = series([100, 101, 102, 103, 104, 110])  # Mon 5 Jan .. Mon 12 Jan
    saturday = pd.Timestamp("2026-01-10")
    # Base is Friday's close (104), not a missing Saturday.
    assert ind.period_return(close, saturday) == pytest.approx(110 / 104 - 1)


def test_period_return_none_when_history_too_short() -> None:
    close = series([100, 110])
    assert ind.period_return(close, pd.Timestamp("2025-01-01")) is None
    assert ind.period_return(pd.Series(dtype=float), pd.Timestamp("2026-01-01")) is None


def test_lookback_starts() -> None:
    starts = ind.lookback_starts(pd.Timestamp("2026-10-07"))
    assert starts["1M"] == pd.Timestamp("2026-09-07")
    assert starts["6M"] == pd.Timestamp("2026-04-07")
    assert starts["1Y"] == pd.Timestamp("2025-10-07")
    assert starts["YTD"] == pd.Timestamp("2025-12-31")  # prior year's last close


# ---------------------------------------------------------------- moving averages


def test_sma_hand_calculated() -> None:
    result = ind.sma(series([1, 2, 3, 4, 5]), 3)
    assert result.iloc[:2].isna().all()
    assert list(result.iloc[2:]) == [2.0, 3.0, 4.0]


def test_latest_sma_insufficient_history() -> None:
    assert ind.latest_sma(series([1, 2]), 3) is None
    assert ind.latest_sma(series([1, 2, 3]), 3) == 2.0


# ---------------------------------------------------------------- RSI


def test_rsi_hand_calculated_wilder() -> None:
    # Closes 10, 11, 12, 11, 13 -> changes +1, +1, -1, +2 (period 3)
    # Seed (first 3 changes): avg gain 2/3, avg loss 1/3 -> RS 2 -> RSI 66.67
    # Next: gain (2/3*2 + 2)/3 = 10/9, loss (1/3*2 + 0)/3 = 2/9 -> RS 5 -> RSI 83.33
    result = ind.rsi(series([10, 11, 12, 11, 13]), period=3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(100 - 100 / 3)
    assert result.iloc[4] == pytest.approx(100 - 100 / 6)


def test_rsi_extremes() -> None:
    assert ind.latest_rsi(series(range(1, 30))) == 100.0  # only gains
    assert ind.latest_rsi(series(range(30, 1, -1))) == pytest.approx(0.0)  # only losses
    assert ind.latest_rsi(series([5.0] * 20)) == 50.0  # no movement


def test_rsi_bounded_on_noisy_data() -> None:
    values = [100 + 5 * math.sin(i / 3) + (i % 7) for i in range(120)]
    rsi = ind.rsi(series(values)).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_rsi_insufficient_history() -> None:
    assert ind.latest_rsi(series([1, 2, 3])) is None


# ---------------------------------------------------------------- volume


def test_relative_volume_excludes_latest_session() -> None:
    avg, ratio = ind.relative_volume(series([100, 200, 300, 400]), lookback=3)
    assert avg == 200.0  # mean of 100, 200, 300
    assert ratio == 2.0  # 400 / 200


def test_relative_volume_insufficient_history() -> None:
    assert ind.relative_volume(series([100, 200]), lookback=3) == (None, None)
