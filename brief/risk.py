"""Risk profile section: volatility, beta, drawdown, Sharpe, and up/down capture.

The metric functions are pure (Series in, number out) and unit-tested against
hand-calculated examples. ``build_risk_profile`` computes every metric for the
ticker and for the benchmark (SPY) so each number can be shown with context.

Conventions: returns are fractions from dividend-adjusted closes (total
return); "1Y" means the last 252 trading days; volatility is annualized with
sqrt(252). Beta and correlation use the full 2-year daily history: one year
proved too unstable (AAPL's correlation with SPY was 0.36 over 1Y vs 0.61
over 2Y). Capture ratios use complete calendar months over the full history.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf

from brief.market_data import SOURCE
from brief.models import DataUnavailableError, SectionResult, safe_section, utc_now
from brief.trends import bar_close_time, fetch_price_history

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_TICKER = "^IRX"  # 13-week US Treasury bill yield, quoted in percent
MIN_CAPTURE_MONTHS = 12


def _clean(value: float | None) -> float | None:
    """NaN/inf -> None."""
    return float(value) if value is not None and math.isfinite(value) else None


# ---------------------------------------------------------------- metrics


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns."""
    return close.pct_change().dropna()


def annualized_volatility(returns: pd.Series) -> float | None:
    """Sample standard deviation of daily returns, scaled to a year."""
    if len(returns) < 2:
        return None
    return _clean(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _aligned(asset: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Asset and benchmark returns on their common dates."""
    return pd.concat({"asset": asset, "bench": benchmark}, axis=1, join="inner").dropna()


def beta(asset: pd.Series, benchmark: pd.Series) -> float | None:
    """Covariance with the benchmark divided by the benchmark's variance."""
    both = _aligned(asset, benchmark)
    if len(both) < 2 or both["bench"].var(ddof=1) == 0:
        return None
    return _clean(both["asset"].cov(both["bench"]) / both["bench"].var(ddof=1))


def correlation(asset: pd.Series, benchmark: pd.Series) -> float | None:
    """Pearson correlation of daily returns."""
    both = _aligned(asset, benchmark)
    return _clean(both["asset"].corr(both["bench"])) if len(both) >= 2 else None


@dataclass(frozen=True)
class Drawdown:
    """Worst peak-to-trough decline in a window."""

    depth: float  # negative fraction, e.g. -0.25
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None  # first close back at the peak, if any


def max_drawdown(close: pd.Series) -> Drawdown | None:
    """Largest fall from a running high, with its dates."""
    close = close.dropna()
    if len(close) < 2:
        return None
    running_peak = close.cummax()
    underwater = close / running_peak - 1
    trough_date = underwater.idxmin()
    depth = float(underwater.loc[trough_date])
    if depth == 0:
        return Drawdown(0.0, close.index[0], close.index[0], close.index[0])
    peak_date = close.loc[:trough_date].idxmax()
    after = close.loc[trough_date:]
    recovered = after[after >= close.loc[peak_date]]
    return Drawdown(depth, peak_date, trough_date, recovered.index[0] if not recovered.empty else None)


def drawdown_series(close: pd.Series) -> pd.Series:
    """Percent below the running high at each date (0 at new highs)."""
    return close / close.cummax() - 1


def sharpe_ratio(period_return: float | None, risk_free: float | None, volatility: float | None) -> float | None:
    """(Return - risk-free rate) / volatility, all annual fractions."""
    if None in (period_return, risk_free, volatility) or not volatility:
        return None
    return _clean((period_return - risk_free) / volatility)


def monthly_returns(close: pd.Series) -> pd.Series:
    """Returns for complete calendar months only (a month in progress is dropped)."""
    month_end = close.resample("ME").last()
    returns = month_end.pct_change().dropna()
    last = close.index[-1]
    if not returns.empty and last < last + pd.offsets.BMonthEnd(0):  # data stops before month's last business day
        returns = returns.iloc[:-1]
    return returns


def _geometric_mean(returns: pd.Series) -> float:
    """Average compounded return per period."""
    return float((1 + returns).prod() ** (1 / len(returns)) - 1)


def capture_ratios(asset_monthly: pd.Series, bench_monthly: pd.Series) -> tuple[float | None, float | None]:
    """Up and down capture vs. the benchmark (1.10 = 110%), Morningstar-style geometric means.

    Up capture uses the months the benchmark rose; down capture the months it fell.
    """
    both = _aligned(asset_monthly, bench_monthly)
    if len(both) < MIN_CAPTURE_MONTHS:
        return None, None

    def ratio(mask: pd.Series) -> float | None:
        months = both[mask]
        if months.empty:
            return None
        bench = _geometric_mean(months["bench"])
        return _clean(_geometric_mean(months["asset"]) / bench) if bench else None

    return ratio(both["bench"] > 0), ratio(both["bench"] < 0)


# ---------------------------------------------------------------- section


@dataclass(frozen=True)
class RiskMetrics:
    """One security's risk numbers. Fractions throughout; ratios as plain floats."""

    volatility_1y: float | None
    return_1y: float | None
    sharpe_1y: float | None
    drawdown_1y: Drawdown | None
    drawdown_2y: Drawdown | None
    beta_2y: float | None
    correlation_2y: float | None
    up_capture: float | None
    down_capture: float | None


@dataclass(frozen=True)
class RiskProfile:
    """Risk section: the ticker's metrics, the benchmark's, and chart data."""

    ticker: RiskMetrics
    benchmark: RiskMetrics | None
    benchmark_symbol: str
    risk_free_rate: float | None
    capture_months: int
    underwater: pd.DataFrame  # drawdown series for ticker (and benchmark)


def compute_metrics(close: pd.Series, bench_close: pd.Series | None, risk_free: float | None) -> RiskMetrics:
    """All risk metrics for one price series, relative to an optional benchmark."""
    returns = daily_returns(close)
    one_year = returns.iloc[-TRADING_DAYS:]
    close_1y = close.iloc[-(TRADING_DAYS + 1):]
    return_1y = _clean(close_1y.iloc[-1] / close_1y.iloc[0] - 1) if len(close) > TRADING_DAYS else None
    vol = annualized_volatility(one_year) if len(returns) >= TRADING_DAYS // 2 else None

    b = corr = up = down = None
    if bench_close is not None and not bench_close.empty:
        bench_returns = daily_returns(bench_close)
        b, corr = beta(returns, bench_returns), correlation(returns, bench_returns)
        up, down = capture_ratios(monthly_returns(close), monthly_returns(bench_close))

    return RiskMetrics(
        volatility_1y=vol,
        return_1y=return_1y,
        sharpe_1y=sharpe_ratio(return_1y, risk_free, vol),
        drawdown_1y=max_drawdown(close_1y),
        drawdown_2y=max_drawdown(close),
        beta_2y=b,
        correlation_2y=corr,
        up_capture=up,
        down_capture=down,
    )


def build_risk_profile(
    history: pd.DataFrame,
    benchmark_history: pd.DataFrame | None,
    benchmark_symbol: str,
    risk_free: float | None,
    symbol: str,
) -> RiskProfile:
    """Risk profile for a ticker, with the benchmark's own metrics for comparison."""
    close = history["Close"].dropna()
    if len(close) < 60:
        raise DataUnavailableError("Not enough price history to measure risk.")
    bench_close = (
        benchmark_history["Close"].dropna()
        if benchmark_history is not None and not benchmark_history.empty
        else None
    )
    is_benchmark = symbol == benchmark_symbol
    ticker_metrics = compute_metrics(close, None if is_benchmark else bench_close, risk_free)
    bench_metrics = (
        compute_metrics(bench_close, bench_close, risk_free)
        if bench_close is not None and not is_benchmark
        else None
    )

    underwater = pd.DataFrame({symbol: drawdown_series(close)})
    if bench_close is not None and not is_benchmark:
        underwater[benchmark_symbol] = drawdown_series(bench_close)
    months = len(_aligned(monthly_returns(close), monthly_returns(bench_close))) if bench_close is not None else 0

    return RiskProfile(
        ticker=ticker_metrics,
        benchmark=bench_metrics,
        benchmark_symbol=benchmark_symbol,
        risk_free_rate=risk_free,
        capture_months=months,
        underwater=underwater,
    )


def fetch_risk_free_rate() -> float | None:
    """Latest 13-week T-bill yield as an annual fraction (4.05% -> 0.0405), or None."""
    try:
        history = yf.Ticker(RISK_FREE_TICKER).history(period="5d")
        rate = _clean(float(history["Close"].dropna().iloc[-1]))
        return rate / 100 if rate is not None else None
    except Exception as exc:  # optional input: Sharpe just shows as unavailable
        logger.warning("Risk-free rate unavailable: %s", exc)
        return None


@safe_section(SOURCE)
def get_risk_profile(
    symbol: str,
    benchmark: str = "SPY",
    history: pd.DataFrame | None = None,
    benchmark_history: pd.DataFrame | None = None,
    risk_free: float | None = None,
) -> SectionResult[RiskProfile]:
    """Risk profile section. A missing benchmark or T-bill rate only blanks the metrics that need it."""
    if history is None:
        history = fetch_price_history(symbol)
    if benchmark_history is None and symbol != benchmark:
        try:
            benchmark_history = fetch_price_history(benchmark)
        except DataUnavailableError as exc:
            logger.warning("Benchmark %s unavailable: %s", benchmark, exc)
    if risk_free is None:
        risk_free = fetch_risk_free_rate()
    profile = build_risk_profile(history, benchmark_history, benchmark, risk_free, symbol)
    as_of: datetime = bar_close_time(history.index[-1]) if len(history) else utc_now()
    return SectionResult.success(profile, SOURCE, as_of=as_of)


def risk_summary(profile: RiskProfile, symbol: str) -> str | None:
    """One plain-English sentence comparing the ticker's risk with the benchmark's.

    Built from rules, not a model, so it can't misstate the numbers.
    """
    t, b = profile.ticker, profile.benchmark
    if b is None or t.volatility_1y is None or not b.volatility_1y:
        return None
    ratio = t.volatility_1y / b.volatility_1y
    if ratio >= 1.15:
        parts = [f"{symbol} has been about {ratio:.1f}x as volatile as {profile.benchmark_symbol} over the past year"]
    elif ratio <= 0.87:
        parts = [f"{symbol} has been calmer than {profile.benchmark_symbol}, at about {ratio:.1f}x its volatility"]
    else:
        parts = [f"{symbol}'s volatility has been similar to {profile.benchmark_symbol}'s"]
    if t.sharpe_1y is not None and b.sharpe_1y is not None:
        better = "better" if t.sharpe_1y > b.sharpe_1y else "weaker"
        parts.append(f"with a {better} risk-adjusted return (Sharpe {t.sharpe_1y:.2f} vs. {b.sharpe_1y:.2f})")
    sentence = ", ".join(parts) + "."
    if t.down_capture is not None and t.up_capture is not None:
        sentence += (
            f" In the market's down months it fell {t.down_capture:.0%} as much as {profile.benchmark_symbol}, "
            f"and in up months it gained {t.up_capture:.0%} as much."
        )
    return sentence
