"""Market data from yfinance: ticker validation, company snapshot, and ETF profile.

Design: thin ``fetch_*`` functions do the network I/O; pure ``parse_*`` functions
turn raw yfinance payloads into typed dataclasses. Parsers are unit-tested
offline against saved fixtures, and the app layer caches the fetchers.

Unit conventions: every ratio leaving this module is a plain fraction
(0.0032 == 0.32%). yfinance is inconsistent here, so conversions happen in
the parsers, not in the UI.
"""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from brief.models import DataUnavailableError, SectionResult, safe_section

SOURCE = "Yahoo Finance (yfinance)"

# 1–6 alphanumerics, optional share-class suffix (BRK-B). Rejects junk early.
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}(-[A-Z]{1,2})?$")


class AssetType(str, Enum):
    """Asset types the brief supports."""

    STOCK = "stock"
    ETF = "etf"


_QUOTE_TYPES = {"EQUITY": AssetType.STOCK, "ETF": AssetType.ETF}
_UNSUPPORTED_NAMES = {
    "MUTUALFUND": "mutual fund",
    "CRYPTOCURRENCY": "cryptocurrency",
    "INDEX": "market index",
    "FUTURE": "futures contract",
    "CURRENCY": "currency pair",
}


class TickerValidationError(ValueError):
    """Raised when a ticker is malformed, unknown, or unsupported."""


class DataSourceError(DataUnavailableError):
    """Raised when an upstream data provider fails or rate-limits us."""


@dataclass(frozen=True)
class TickerInfo:
    """Identity of a validated ticker."""

    symbol: str
    name: str
    asset_type: AssetType
    exchange: str | None
    currency: str


@dataclass(frozen=True)
class Snapshot:
    """Headline quote and profile data. Ratios are fractions."""

    price: float | None
    day_change: float | None
    day_change_pct: float | None
    market_cap: float | None
    week52_low: float | None
    week52_high: float | None
    dividend_yield: float | None
    sector: str | None
    industry: str | None


@dataclass(frozen=True)
class Holding:
    """One ETF constituent."""

    symbol: str
    name: str
    weight: float  # fraction of fund assets


@dataclass(frozen=True)
class EtfProfile:
    """ETF-specific facts shown instead of company financials."""

    expense_ratio: float | None  # fraction
    aum: float | None  # total net assets, USD
    category: str | None
    fund_family: str | None
    top_holdings: list[Holding]


# ---------------------------------------------------------------- helpers


def first_number(info: dict[str, Any], *keys: str) -> float | None:
    """Return the first present, finite numeric value among ``keys``."""
    for key in keys:
        value = info.get(key)
        # numbers.Real also covers numpy scalars (int64, float64) from pandas tables.
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            if math.isfinite(value):
                return float(value)
    return None


def _pct_to_fraction(value: float | None) -> float | None:
    """Convert a percent-unit number (0.32 meaning 0.32%) to a fraction."""
    return None if value is None else value / 100


def quote_time(info: dict[str, Any]) -> datetime | None:
    """Timestamp of the last market quote, if yfinance reports one."""
    ts = first_number(info, "regularMarketTime")
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None


# ---------------------------------------------------------------- validation


def normalize_symbol(raw: str) -> str:
    """Clean user input into a Yahoo-style symbol (e.g. ' brk.b ' -> 'BRK-B')."""
    symbol = raw.strip().upper().lstrip("$")
    # Share classes use a dash on Yahoo; only rewrite single-letter suffixes so
    # foreign suffixes like SHOP.TO are left alone (and rejected below).
    symbol = re.sub(r"\.([A-Z])$", r"-\1", symbol)
    if not _SYMBOL_RE.match(symbol):
        raise TickerValidationError(
            f"'{raw.strip()}' doesn't look like a US ticker symbol."
        )
    return symbol


def fetch_info(symbol: str) -> dict[str, Any]:
    """Fetch yfinance's quote/profile dict for ``symbol``."""
    try:
        return yf.Ticker(symbol).info or {}
    except YFRateLimitError as exc:
        raise DataSourceError(
            "Yahoo Finance is rate-limiting requests. Try again in a minute."
        ) from exc
    except Exception as exc:  # yfinance raises bare HTTP errors for some failures
        # Unknown symbols sometimes raise instead of returning an empty dict.
        if "404" in str(exc) or "Not Found" in str(exc):
            return {}
        raise DataSourceError(f"Yahoo Finance request failed: {exc}") from exc


def parse_ticker_info(symbol: str, info: dict[str, Any]) -> TickerInfo:
    """Validate a raw info payload and extract the ticker's identity."""
    quote_type = info.get("quoteType")
    if not quote_type or first_number(info, "regularMarketPrice", "currentPrice") is None:
        raise TickerValidationError(f"No security found for '{symbol}'.")
    if quote_type not in _QUOTE_TYPES:
        raise TickerValidationError(
            f"{symbol} is a {_UNSUPPORTED_NAMES.get(quote_type, quote_type.lower())}; "
            "only stocks and ETFs are supported."
        )
    currency = info.get("currency") or ""
    if currency != "USD":
        raise TickerValidationError(
            f"{symbol} trades in {currency or 'an unknown currency'}; "
            "only US-listed securities are supported."
        )
    return TickerInfo(
        symbol=symbol,
        name=info.get("longName") or info.get("shortName") or symbol,
        asset_type=_QUOTE_TYPES[quote_type],
        exchange=info.get("exchange"),
        currency=currency,
    )


def validate_ticker(raw: str) -> tuple[TickerInfo, dict[str, Any]]:
    """Normalize, fetch, and validate a ticker. Returns identity plus raw info.

    The raw info is returned so callers can build sections without a second
    network round trip.
    """
    symbol = normalize_symbol(raw)
    info = fetch_info(symbol)
    return parse_ticker_info(symbol, info), info


# ---------------------------------------------------------------- snapshot


def parse_snapshot(info: dict[str, Any]) -> Snapshot:
    """Build a Snapshot from a raw info payload; missing fields become None."""
    price = first_number(info, "currentPrice", "regularMarketPrice")
    prev_close = first_number(info, "regularMarketPreviousClose", "previousClose")

    change = first_number(info, "regularMarketChange")
    if change is None and price is not None and prev_close:
        change = price - prev_close
    change_pct = _pct_to_fraction(first_number(info, "regularMarketChangePercent"))
    if change_pct is None and change is not None and prev_close:
        change_pct = change / prev_close

    # ``dividendYield`` is percent units; the trailing field is already a fraction.
    dividend_yield = _pct_to_fraction(first_number(info, "dividendYield"))
    if dividend_yield is None:
        dividend_yield = first_number(info, "trailingAnnualDividendYield")

    return Snapshot(
        price=price,
        day_change=change,
        day_change_pct=change_pct,
        market_cap=first_number(info, "marketCap"),
        week52_low=first_number(info, "fiftyTwoWeekLow"),
        week52_high=first_number(info, "fiftyTwoWeekHigh"),
        dividend_yield=dividend_yield,
        sector=info.get("sector"),
        industry=info.get("industry"),
    )


@safe_section(SOURCE)
def get_snapshot(info: dict[str, Any]) -> SectionResult[Snapshot]:
    """Snapshot section, stamped with the time of the last quote."""
    return SectionResult.success(parse_snapshot(info), SOURCE, as_of=quote_time(info))


# ---------------------------------------------------------------- ETF profile


def fetch_top_holdings(symbol: str) -> pd.DataFrame:
    """Fetch an ETF's top holdings table from yfinance."""
    try:
        return yf.Ticker(symbol).funds_data.top_holdings
    except YFRateLimitError as exc:
        raise DataSourceError(
            "Yahoo Finance is rate-limiting requests. Try again in a minute."
        ) from exc


def parse_top_holdings(df: pd.DataFrame | None) -> list[Holding]:
    """Convert yfinance's holdings table (index=symbol) into Holding objects."""
    if df is None or df.empty:
        return []
    table = df.reset_index()
    symbol_col, name_col = table.columns[0], "Name"
    weight_col = next(c for c in table.columns if "percent" in str(c).lower())
    return [
        Holding(symbol=str(row[symbol_col]), name=str(row[name_col]), weight=float(row[weight_col]))
        for _, row in table.iterrows()
        if pd.notna(row[weight_col])
    ]


def parse_etf_profile(info: dict[str, Any], holdings: list[Holding]) -> EtfProfile:
    """Combine info fields and holdings into an EtfProfile."""
    return EtfProfile(
        # ``netExpenseRatio`` is percent units (0.0945 == 0.0945%).
        expense_ratio=_pct_to_fraction(first_number(info, "netExpenseRatio")),
        aum=first_number(info, "totalAssets"),
        category=info.get("category"),
        fund_family=info.get("fundFamily"),
        top_holdings=holdings,
    )


@safe_section(SOURCE)
def get_etf_profile(symbol: str, info: dict[str, Any]) -> SectionResult[EtfProfile]:
    """ETF profile section: expense ratio, AUM, and top holdings."""
    holdings = parse_top_holdings(fetch_top_holdings(symbol))
    return SectionResult.success(parse_etf_profile(info, holdings), SOURCE)
