"""Cached data loaders for the Streamlit app.

The data modules know nothing about Streamlit; caching lives here. Every loader
is keyed by plain strings (the ticker), and only successful sections are
cached, so a transient failure (rate limit, timeout) is retried on the next
lookup instead of being served from cache for 15 minutes.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import pandas as pd
import streamlit as st

from brief.ai_analysis import AnalysisResult, get_ai_analysis
from brief.config import get_settings
from brief.financials import Financials, get_financials
from brief.finnhub_client import (
    EarningsEvent,
    EarningsHistory,
    WallStreetView,
    get_earnings_history,
    get_next_earnings,
    get_wall_street_view,
)
from brief.market_data import AssetType, EtfProfile, Snapshot, TickerInfo, get_etf_profile, get_snapshot, validate_ticker
from brief.models import DataUnavailableError, SectionResult
from brief.news import NewsBrief, get_news
from brief.ownership import OwnershipActivity, get_ownership_activity
from brief.risk import RiskProfile, fetch_risk_free_rate, get_risk_profile
from brief.symbols import SymbolEntry, fallback_universe, fetch_universe
from brief.trends import PriceTrends, fetch_price_history, get_price_trends

TTL = get_settings().cache_ttl_seconds
F = TypeVar("F", bound=Callable[..., SectionResult[Any]])


class _FailedResult(Exception):
    """Carries a failed SectionResult out of the cache (exceptions aren't cached)."""

    def __init__(self, result: SectionResult[Any]) -> None:
        super().__init__(result.error)
        self.result = result


def cache_successes(func: F) -> F:
    """Cache a section loader for TTL seconds, but never cache a failed result."""

    def _cached(*args: Any) -> SectionResult[Any]:
        result = func(*args)
        if not result.ok:
            raise _FailedResult(result)
        return result

    # Streamlit keys caches by function name; give each wrapper a unique one.
    _cached.__name__ = _cached.__qualname__ = f"{func.__qualname__}__cached"
    cached = st.cache_data(ttl=TTL, show_spinner=False)(_cached)

    @functools.wraps(func)
    def wrapper(*args: Any) -> SectionResult[Any]:
        try:
            return cached(*args)
        except _FailedResult as exc:
            return exc.result

    return wrapper  # type: ignore[return-value]


@st.cache_data(ttl=TTL, show_spinner=False)
def load_ticker(symbol: str) -> tuple[TickerInfo, dict[str, Any]]:
    """Validated ticker identity and raw quote info. Errors propagate (and aren't cached)."""
    return validate_ticker(symbol)


@cache_successes
def load_snapshot(symbol: str) -> SectionResult[Snapshot]:
    """Company snapshot."""
    return get_snapshot(load_ticker(symbol)[1])


@cache_successes
def load_etf_profile(symbol: str) -> SectionResult[EtfProfile]:
    """ETF holdings, expense ratio, and AUM."""
    return get_etf_profile(symbol, load_ticker(symbol)[1])


@cache_successes
def load_financials(symbol: str) -> SectionResult[Financials]:
    """Quarterly financials and valuation ratios."""
    return get_financials(symbol, load_ticker(symbol)[1])


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _cached_universe() -> list[SymbolEntry]:
    return fetch_universe()  # raises (uncached) on failure


def load_universe() -> list[SymbolEntry]:
    """Searchable tickers and company names (refreshed daily); the curated list if Finnhub is down."""
    try:
        return _cached_universe()
    except Exception:  # search must always work, even without the full list
        return fallback_universe()


@st.cache_data(ttl=TTL, show_spinner=False)
def _cached_history(symbol: str) -> pd.DataFrame:
    """Two years of daily bars; raises (uncached) on failure."""
    return fetch_price_history(symbol)


def load_history(symbol: str) -> pd.DataFrame | None:
    """Shared price history for the trends and risk sections, or None if unavailable.

    On None, the section fetches again itself and reports the error properly.
    """
    try:
        return _cached_history(symbol)
    except DataUnavailableError:
        return None


@st.cache_data(ttl=TTL, show_spinner=False)
def _cached_risk_free() -> float:
    rate = fetch_risk_free_rate()
    if rate is None:
        raise DataUnavailableError("Risk-free rate unavailable")  # don't cache a miss
    return rate


def load_risk_free() -> float | None:
    """13-week T-bill yield as a fraction, or None (Sharpe then shows as unavailable)."""
    try:
        return _cached_risk_free()
    except DataUnavailableError:
        return None


@cache_successes
def load_trends(symbol: str) -> SectionResult[PriceTrends]:
    """Price trends vs. the benchmark."""
    benchmark = get_settings().benchmark_ticker
    return get_price_trends(symbol, benchmark, load_history(symbol), load_history(benchmark))


@cache_successes
def load_risk(symbol: str) -> SectionResult[RiskProfile]:
    """Risk profile vs. the benchmark, reusing the cached price history."""
    benchmark = get_settings().benchmark_ticker
    return get_risk_profile(symbol, benchmark, load_history(symbol), load_history(benchmark), load_risk_free())


@cache_successes
def load_earnings(symbol: str) -> SectionResult[EarningsEvent]:
    """Next earnings date."""
    return get_next_earnings(symbol)


@cache_successes
def load_earnings_history(symbol: str) -> SectionResult[EarningsHistory]:
    """Last four quarters of EPS vs. consensus."""
    return get_earnings_history(symbol)


@cache_successes
def load_ownership(symbol: str) -> SectionResult[OwnershipActivity]:
    """Insider trades (6 months), ownership split, and top institutions."""
    return get_ownership_activity(symbol)


@cache_successes
def load_wall_street(symbol: str) -> SectionResult[WallStreetView]:
    """Analyst consensus and price target."""
    return get_wall_street_view(symbol, load_ticker(symbol)[1])


@cache_successes
def load_news(symbol: str) -> SectionResult[NewsBrief]:
    """Recent headlines with sentiment."""
    ticker = load_ticker(symbol)[0]
    return get_news(symbol, ticker.name, is_etf=ticker.asset_type is AssetType.ETF)


@cache_successes
def load_analysis(symbol: str) -> SectionResult[AnalysisResult]:
    """AI analysis over the other sections (all of which are cache hits by now)."""
    ticker = load_ticker(symbol)[0]
    is_etf = ticker.asset_type is AssetType.ETF
    return get_ai_analysis(
        ticker,
        snapshot=load_snapshot(symbol),
        financials=None if is_etf else load_financials(symbol),
        etf=load_etf_profile(symbol) if is_etf else None,
        trends=load_trends(symbol),
        risk=load_risk(symbol),
        wall_street=None if is_etf else load_wall_street(symbol),
        earnings=None if is_etf else load_earnings(symbol),
        earnings_history=None if is_etf else load_earnings_history(symbol),
        ownership=None if is_etf else load_ownership(symbol),
        news=load_news(symbol),
    )
