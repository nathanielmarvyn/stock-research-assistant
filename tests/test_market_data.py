"""Parsing tests for brief.market_data, run offline against captured payloads."""

import pandas as pd
import pytest

from brief.market_data import (
    AssetType,
    TickerValidationError,
    normalize_symbol,
    parse_etf_profile,
    parse_snapshot,
    parse_ticker_info,
    parse_top_holdings,
    quote_time,
)


@pytest.mark.parametrize(
    "raw, expected",
    [("aapl", "AAPL"), (" $msft ", "MSFT"), ("brk.b", "BRK-B"), ("BRK-B", "BRK-B")],
)
def test_normalize_symbol(raw: str, expected: str) -> None:
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["", "123", "TOOLONGTICKER", "SHOP.TO", "AA PL"])
def test_normalize_symbol_rejects_junk(raw: str) -> None:
    with pytest.raises(TickerValidationError):
        normalize_symbol(raw)


def test_stock_identity(aapl_info: dict) -> None:
    info = parse_ticker_info("AAPL", aapl_info)
    assert info.name == "Apple Inc."
    assert info.asset_type is AssetType.STOCK


def test_etf_identity(spy_info: dict) -> None:
    assert parse_ticker_info("SPY", spy_info).asset_type is AssetType.ETF


def test_unknown_ticker_rejected(unknown_info: dict) -> None:
    with pytest.raises(TickerValidationError, match="No security found"):
        parse_ticker_info("ZZZZQ", unknown_info)


def test_unsupported_type_rejected() -> None:
    info = {"quoteType": "MUTUALFUND", "regularMarketPrice": 10.0, "currency": "USD"}
    with pytest.raises(TickerValidationError, match="only stocks and ETFs"):
        parse_ticker_info("VFIAX", info)


def test_non_usd_rejected() -> None:
    info = {"quoteType": "EQUITY", "regularMarketPrice": 10.0, "currency": "CAD"}
    with pytest.raises(TickerValidationError, match="US-listed"):
        parse_ticker_info("XYZ", info)


def test_snapshot_converts_percent_units(aapl_info: dict) -> None:
    snap = parse_snapshot(aapl_info)
    # yfinance reports dividendYield=0.32 meaning 0.32%; we store a fraction.
    assert snap.dividend_yield == pytest.approx(0.0032)
    assert snap.day_change_pct == pytest.approx(aapl_info["regularMarketChangePercent"] / 100)
    assert snap.week52_low <= snap.price <= snap.week52_high
    assert snap.sector == "Technology"


def test_etf_snapshot_uses_market_price(spy_info: dict) -> None:
    snap = parse_snapshot(spy_info)  # ETFs have no currentPrice field
    assert snap.price == spy_info["regularMarketPrice"]
    assert snap.sector is None


def test_snapshot_tolerates_missing_fields() -> None:
    snap = parse_snapshot({"regularMarketPrice": 100.0, "regularMarketPreviousClose": 98.0})
    assert snap.day_change == pytest.approx(2.0)
    assert snap.day_change_pct == pytest.approx(2.0 / 98.0)
    assert snap.market_cap is None and snap.dividend_yield is None


def test_snapshot_ignores_non_finite_values() -> None:
    assert parse_snapshot({"marketCap": float("nan")}).market_cap is None


def test_quote_time_is_utc(aapl_info: dict) -> None:
    assert quote_time(aapl_info).tzinfo is not None
    assert quote_time({}) is None


def test_etf_profile(spy_info: dict, spy_holdings_df: pd.DataFrame) -> None:
    holdings = parse_top_holdings(spy_holdings_df)
    profile = parse_etf_profile(spy_info, holdings)
    assert profile.expense_ratio == pytest.approx(0.000945)  # 0.0945%
    assert profile.aum and profile.aum > 1e11
    assert holdings[0].symbol == "NVDA"
    assert 0 < holdings[0].weight < 1


def test_empty_holdings() -> None:
    assert parse_top_holdings(pd.DataFrame()) == []
    assert parse_top_holdings(None) == []
