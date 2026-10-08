"""Tests for UI helpers and chart builders (no Streamlit server needed)."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from brief.financials import parse_financials
from brief.finnhub_client import AnalystConsensus
from brief.market_data import Holding
from brief.models import SectionResult
from brief.news import _first_sentence
from brief.ui import charts
from brief.ui import sections as ui


def test_md_escapes_dollars_for_latex() -> None:
    assert ui.md("$243.42 – $345.34") == r"\$243.42 – \$345.34"


def test_date_only_label_is_not_shifted_by_timezone() -> None:
    result = SectionResult.success(1, "t", as_of=datetime(2026, 6, 30, tzinfo=timezone.utc))
    assert ui.as_of_text(result, date_only=True) == "Jun 30, 2026"  # not Jun 29 in New York


def test_timestamp_label_in_eastern_time() -> None:
    result = SectionResult.success(1, "t", as_of=datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc))
    assert ui.as_of_text(result) == "Oct 7, 2026, 4:00 PM ET"


@pytest.mark.parametrize(
    "low, high, price, expected",
    [(100, 200, 150, 0.5), (100, 200, 250, 1.0), (100, 100, 100, None), (None, 200, 150, None)],
)
def test_range_position(low, high, price, expected) -> None:
    assert ui.range_position(low, high, price) == expected


@pytest.mark.parametrize("rsi, zone", [(75, "Overbought (>70)"), (25, "Oversold (<30)"), (50, "Neutral (30–70)"), (None, "")])
def test_rsi_zone(rsi, zone) -> None:
    assert ui.rsi_zone(rsi) == zone


def test_financials_table_shape(aapl_statements, aapl_info) -> None:
    table = ui.financials_table(parse_financials(*aapl_statements, aapl_info))
    assert table.shape == (11, 4)
    assert table.columns[0] == "Qtr ended Jun 30, 2026"
    assert table.loc["Revenue"].iloc[0].startswith("$")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A U.S. court ordered Apple to pay $184M in interest on patent damages. More text.",
         "A U.S. court ordered Apple to pay $184M in interest on patent damages."),
        ("Micron reported $37.5 billion in revenue this quarter, beating estimates. Shares rose.",
         "Micron reported $37.5 billion in revenue this quarter, beating estimates."),
        ("For JPMorgan's Q3 2026 earnings release on Oct. 14, analysts expect growth. Shares rose.",
         "For JPMorgan's Q3 2026 earnings release on Oct. 14, analysts expect growth."),
        ("Short.", "Short."),
    ],
)
def test_first_sentence_handles_abbreviations(text: str, expected: str) -> None:
    assert _first_sentence(text) == expected


def test_price_chart_uses_separate_volume_panel() -> None:
    idx = pd.bdate_range("2026-01-01", periods=30)
    chart = pd.DataFrame({"Close": range(30), "SMA50": [None] * 30, "SMA200": [None] * 30, "Volume": [1e6] * 30}, index=idx)
    fig = charts.price_chart(chart, "AAPL")
    names = [t.name for t in fig.data]
    assert names == ["AAPL", "Volume"]  # empty SMAs are omitted
    assert fig.data[1].yaxis == "y2"  # volume on its own panel, not a second y-axis on the price panel


def test_ratings_chart_has_all_five_buckets() -> None:
    c = AnalystConsensus(date(2026, 9, 1), 12, 22, 15, 3, 1, 2.23, "Buy", 2.13)
    fig = charts.ratings_chart(c)
    assert [t.name for t in fig.data] == ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    assert [t.x[0] for t in fig.data] == [12, 22, 15, 3, 1]


def test_holdings_chart_largest_on_top() -> None:
    fig = charts.holdings_chart([Holding("NVDA", "Nvidia", 0.08), Holding("AAPL", "Apple", 0.07)])
    assert list(fig.data[0].y) == ["AAPL", "NVDA"]  # plotly draws bottom-up


def _risk_profile(with_benchmark: bool = True):
    import math

    from brief.risk import build_risk_profile

    idx = pd.bdate_range("2024-10-01", periods=504, tz="America/New_York")
    asset = pd.DataFrame({"Close": [100 * (1 + 0.02 * math.sin(i / 5)) * 1.0006**i for i in range(504)]}, index=idx)
    bench = pd.DataFrame({"Close": [100 * (1 + 0.01 * math.sin(i / 5)) * 1.0003**i for i in range(504)]}, index=idx)
    return build_risk_profile(asset, bench if with_benchmark else None, "SPY", 0.0405, "XYZ")


def test_risk_table_with_benchmark() -> None:
    table = ui.risk_table(_risk_profile(), "XYZ")
    assert list(table.columns) == ["Metric", "XYZ", "SPY", "What it means"]
    assert {"Beta (2Y)", "Down capture"} <= set(table["Metric"])
    assert "4.05% T-bill" in table.loc[table["Metric"] == "Sharpe ratio (1Y)", "What it means"].iloc[0]


def test_risk_table_without_benchmark_drops_relative_rows() -> None:
    table = ui.risk_table(_risk_profile(with_benchmark=False), "XYZ")
    assert list(table.columns) == ["Metric", "XYZ", "What it means"]
    assert "Beta (2Y)" not in set(table["Metric"])


def test_drawdown_detail() -> None:
    from brief.risk import Drawdown

    d = Drawdown(-0.334, pd.Timestamp("2024-12-26"), pd.Timestamp("2025-04-08"), None)
    assert ui.drawdown_detail(d) == "Worst fall from a high: Dec 26, 2024 to Apr 8, 2025, not yet recovered."


def test_drawdown_chart_fills_ticker_only() -> None:
    fig = charts.drawdown_chart(_risk_profile().underwater)
    assert [t.name for t in fig.data] == ["XYZ", "SPY"]
    assert fig.data[0].fill == "tozeroy" and fig.data[1].fill is None
