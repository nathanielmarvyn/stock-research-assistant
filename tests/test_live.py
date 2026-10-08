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
