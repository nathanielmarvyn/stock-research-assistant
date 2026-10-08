"""Tests for the search universe: name cleanup, ordering, and query resolution."""

import pytest

from brief import symbols as sym
from tests.conftest import load_json


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("APPLE INC", "Apple Inc"),
        ("BERKSHIRE HATHAWAY INC-CL B", "Berkshire Hathaway Inc Class B"),
        ("META PLATFORMS INC-CLASS A", "Meta Platforms Inc Class A"),
        ("VISA INC-CLASS A SHARES", "Visa Inc Class A"),
        ("TAIWAN SEMICONDUCTOR-SP ADR", "Taiwan Semiconductor (ADR)"),
        ("COCA-COLA CO/THE", "Coca-Cola Co"),
        ("SS SPDR S&P 500 ETF TRUST-US", "SPDR S&P 500 ETF Trust"),
        ("INVESCO QQQ TRUST SERIES 1", "Invesco QQQ Trust Series 1"),
        ("AT&T INC", "AT&T Inc"),
        ("ISHARES JP MORGAN USD EMERGI", "iShares JP Morgan USD Emergi"),
    ],
)
def test_clean_name(raw: str, clean: str) -> None:
    assert sym.clean_name(raw) == clean


@pytest.fixture
def universe() -> list[sym.SymbolEntry]:
    return sym.build_universe(load_json("finnhub_us_symbols_sample.json"))


def test_universe_puts_popular_first_and_filters(universe) -> None:
    symbols = [e.symbol for e in universe]
    assert symbols[0] == "AAPL"
    assert "APLE" in symbols and symbols.index("APLE") > symbols.index("AAPL")  # REITs kept, after popular
    assert "MTSFF" not in symbols  # over-the-counter excluded
    assert len(symbols) == len(set(symbols))


def test_entry_kinds_and_labels(universe) -> None:
    by = {e.symbol: e for e in universe}
    assert by["SPY"].kind == "ETF" and by["TSM"].kind == "ADR" and by["APLE"].kind == "REIT"
    assert by["AAPL"].label == "AAPL · Apple Inc."


@pytest.mark.parametrize(
    "query, symbol",
    [
        ("AAPL · Apple Inc.", "AAPL"),  # picked from the dropdown
        ("aapl", "AAPL"),  # ticker, any case
        ("brk-b", "BRK.B"),  # either share-class separator
        ("apple", "AAPL"),  # company name: Apple Inc. before Apple Hospitality
        ("hospitality", "APLE"),
        ("coca-cola", "KO"),
        ("berkshire", "BRK.B"),
        ("jp morgan", "JPM"),  # spaces ignored when matching names
        ("newco", "NEWCO"),  # unknown input passed through as a ticker to try
        ("  ", None),
    ],
)
def test_resolve_query(universe, query: str, symbol: str | None) -> None:
    assert sym.resolve_query(query, universe) == symbol


def test_fallback_universe_is_curated_list() -> None:
    fallback = sym.fallback_universe()
    assert len(fallback) == len(sym.POPULAR) and fallback[0].label == "AAPL · Apple Inc."
