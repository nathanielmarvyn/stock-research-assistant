"""Live smoke tests against real APIs. Run with: pytest -m live"""

import pytest

from brief.market_data import (
    AssetType,
    TickerValidationError,
    get_etf_profile,
    get_snapshot,
    validate_ticker,
)

pytestmark = pytest.mark.live


def test_live_stock_aapl() -> None:
    ticker, info = validate_ticker("aapl")
    assert ticker.asset_type is AssetType.STOCK
    result = get_snapshot(info)
    assert result.ok and result.data.price > 0


def test_live_etf_spy() -> None:
    ticker, info = validate_ticker("SPY")
    assert ticker.asset_type is AssetType.ETF
    profile = get_etf_profile("SPY", info)
    assert profile.ok and len(profile.data.top_holdings) > 0


def test_live_invalid_ticker() -> None:
    with pytest.raises(TickerValidationError):
        validate_ticker("ZZZZQ")


def test_live_financials_aapl() -> None:
    from brief.financials import get_financials

    _, info = validate_ticker("AAPL")
    result = get_financials("AAPL", info)
    assert result.ok and result.data.quarters[0].revenue > 0


@pytest.mark.parametrize("symbol", ["AAPL", "SPY"])
def test_live_price_trends(symbol: str) -> None:
    from brief.trends import get_price_trends

    result = get_price_trends(symbol)
    assert result.ok
    assert result.data.sma200 is not None and result.data.rsi14 is not None
    assert all(p.benchmark_return is not None for p in result.data.performance)
