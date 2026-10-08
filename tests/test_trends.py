"""Tests for assembling the price-trends section from daily history."""

import pandas as pd
import pytest

from brief.trends import build_price_trends


def history(closes: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes), tz="America/New_York")
    return pd.DataFrame({"Close": closes, "Volume": [1_000_000.0] * len(closes)}, index=idx)


@pytest.fixture
def rising() -> pd.DataFrame:
    """~2 years (matching the live fetch) of steadily rising prices, +0.1% per day."""
    return history([100 * 1.001**i for i in range(504)])


def test_trends_with_flat_benchmark(rising: pd.DataFrame) -> None:
    flat = history([50.0] * 504)
    t = build_price_trends(rising, flat, "SPY")
    perf = {p.period: p for p in t.performance}
    assert perf["1Y"].ticker_return > 0.2
    assert perf["1Y"].benchmark_return == 0
    assert perf["1Y"].excess_return == pytest.approx(perf["1Y"].ticker_return)
    assert t.above_sma50 and t.above_sma200
    assert t.rsi14 == 100.0  # never a down day
    assert t.relative_volume == pytest.approx(1.0)


def test_chart_limited_to_one_year(rising: pd.DataFrame) -> None:
    t = build_price_trends(rising, None, "SPY")
    span = t.chart.index[-1] - t.chart.index[0]
    assert span <= pd.Timedelta(days=365)
    assert t.chart["SMA200"].notna().all()  # 200-day history exists before the chart starts


def test_missing_benchmark_blanks_comparison_only(rising: pd.DataFrame) -> None:
    t = build_price_trends(rising, None, "SPY")
    assert all(p.benchmark_return is None and p.excess_return is None for p in t.performance)
    assert all(p.ticker_return is not None for p in t.performance)


def test_short_history_leaves_long_indicators_empty() -> None:
    t = build_price_trends(history([100.0 + i for i in range(60)]), None, "SPY")
    assert t.sma50 is not None and t.sma200 is None
    perf = {p.period: p for p in t.performance}
    assert perf["1Y"].ticker_return is None


def test_bar_close_time_is_4pm_new_york_in_utc() -> None:
    from brief.trends import bar_close_time

    bar = pd.Timestamp("2026-10-07", tz="America/New_York")
    assert bar_close_time(bar) == pd.Timestamp("2026-10-07 20:00", tz="UTC")  # EDT = UTC-4
