"""Price trends section: performance vs. the S&P 500, moving averages, RSI, volume.

``fetch_price_history`` does the I/O; ``build_price_trends`` is pure and
combines indicator functions into one typed result for the UI and the AI prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from brief import indicators as ind
from brief.market_data import SOURCE, DataSourceError
from brief.models import DataUnavailableError, SectionResult, safe_section, utc_now

logger = logging.getLogger(__name__)

# 2 years of daily bars: enough for a 1Y window plus a 200-day SMA across the chart.
HISTORY_PERIOD = "2y"
CHART_DAYS = 365
PERIODS = ("1M", "6M", "YTD", "1Y")


@dataclass(frozen=True)
class PeriodPerformance:
    """Total return over one window for the ticker and the benchmark."""

    period: str
    ticker_return: float | None
    benchmark_return: float | None

    @property
    def excess_return(self) -> float | None:
        """Ticker minus benchmark, in fraction terms (0.02 == 2 percentage points)."""
        if self.ticker_return is None or self.benchmark_return is None:
            return None
        return self.ticker_return - self.benchmark_return


@dataclass(frozen=True)
class PriceTrends:
    """Everything the price-trends section shows, plus chart data."""

    last_close: float | None
    performance: list[PeriodPerformance]
    benchmark: str
    sma50: float | None
    sma200: float | None
    rsi14: float | None
    latest_volume: float | None
    avg_volume_30d: float | None
    relative_volume: float | None
    chart: pd.DataFrame  # last ~1Y: Close, SMA50, SMA200, Volume

    @property
    def above_sma50(self) -> bool | None:
        """Whether the last close is above the 50-day average."""
        if self.last_close is None or self.sma50 is None:
            return None
        return self.last_close > self.sma50

    @property
    def above_sma200(self) -> bool | None:
        """Whether the last close is above the 200-day average."""
        if self.last_close is None or self.sma200 is None:
            return None
        return self.last_close > self.sma200


def fetch_price_history(symbol: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Daily OHLCV history (split- and dividend-adjusted) from yfinance."""
    try:
        history = yf.Ticker(symbol).history(period=period, interval="1d")
    except YFRateLimitError as exc:
        raise DataSourceError(
            "Yahoo Finance is rate-limiting requests. Try again in a minute."
        ) from exc
    if history is None or history.empty:
        raise DataUnavailableError(f"No price history available for {symbol}.")
    return history


def build_price_trends(
    history: pd.DataFrame, benchmark_history: pd.DataFrame | None, benchmark: str
) -> PriceTrends:
    """Compute all trend metrics from daily history. Benchmark data is optional."""
    close = history["Close"].dropna()
    if close.empty:
        raise DataUnavailableError("Price history contains no closing prices.")
    volume = history["Volume"].dropna()
    bench_close = (
        benchmark_history["Close"].dropna()
        if benchmark_history is not None and not benchmark_history.empty
        else pd.Series(dtype=float)
    )

    starts = ind.lookback_starts(close.index[-1])
    performance = [
        PeriodPerformance(
            period=p,
            ticker_return=ind.period_return(close, starts[p]),
            benchmark_return=ind.period_return(bench_close, starts[p]) if not bench_close.empty else None,
        )
        for p in PERIODS
    ]

    avg_volume, rel_volume = ind.relative_volume(volume)

    chart = pd.DataFrame(
        {
            "Close": close,
            "SMA50": ind.sma(close, 50),
            "SMA200": ind.sma(close, 200),
            "Volume": history["Volume"],
        }
    )
    chart = chart.loc[chart.index >= close.index[-1] - pd.Timedelta(days=CHART_DAYS)]

    return PriceTrends(
        last_close=float(close.iloc[-1]),
        performance=performance,
        benchmark=benchmark,
        sma50=ind.latest_sma(close, 50),
        sma200=ind.latest_sma(close, 200),
        rsi14=ind.latest_rsi(close),
        latest_volume=float(volume.iloc[-1]) if not volume.empty else None,
        avg_volume_30d=avg_volume,
        relative_volume=rel_volume,
        chart=chart,
    )


@safe_section(SOURCE)
def get_price_trends(
    symbol: str,
    benchmark: str = "SPY",
    history: pd.DataFrame | None = None,
    benchmark_history: pd.DataFrame | None = None,
) -> SectionResult[PriceTrends]:
    """Price trends section, stamped with the last bar's date.

    Histories can be passed in (e.g. from a cache). A failed benchmark fetch
    only blanks the comparison columns; it doesn't fail the section.
    """
    if history is None:
        history = fetch_price_history(symbol)
    if benchmark_history is None:
        try:
            benchmark_history = fetch_price_history(benchmark)
        except DataUnavailableError as exc:
            logger.warning("Benchmark %s unavailable: %s", benchmark, exc)
    trends = build_price_trends(history, benchmark_history, benchmark)
    return SectionResult.success(trends, SOURCE, as_of=bar_close_time(history.index[-1]))


def bar_close_time(bar_date: pd.Timestamp) -> datetime:
    """UTC time a daily bar's data is current to: 4pm New York, or now if the session is live."""
    day = bar_date.tz_localize("America/New_York") if bar_date.tzinfo is None else bar_date
    close = day.tz_convert("America/New_York").normalize() + pd.Timedelta(hours=16)
    return min(close.tz_convert("UTC").to_pydatetime(), utc_now())
