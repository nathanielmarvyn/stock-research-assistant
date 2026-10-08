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
