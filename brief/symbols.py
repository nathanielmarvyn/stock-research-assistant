"""Search universe: every US-listed stock and ETF, searchable by ticker or company name.

Finnhub's free symbol list (~31,000 US symbols) is filtered to stocks, ETFs,
ADRs, and REITs on the major exchanges (~11,000), and its all-caps names are
tidied for display ("BERKSHIRE HATHAWAY INC-CL B" -> "Berkshire Hathaway Inc
Class B"). Well-known companies and funds are listed first with hand-written
names, because the search box keeps this order when it filters: typing
"apple" should offer Apple Inc. before Apple Hospitality REIT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from brief.finnhub_client import FinnhubClient

LABEL_SEPARATOR = " · "
MAJOR_EXCHANGES = {"XNYS", "XNAS", "ARCX", "BATS", "XASE"}
SEARCHABLE_TYPES = {"Common Stock", "ETP", "ADR", "REIT"}

# Shown first, in this order, with clean names. Roughly the largest US companies
# plus the most-held ETFs.
POPULAR: tuple[tuple[str, str], ...] = (
    ("AAPL", "Apple Inc."), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("AMZN", "Amazon.com"),
    ("GOOGL", "Alphabet (Google) Class A"), ("GOOG", "Alphabet (Google) Class C"), ("META", "Meta Platforms"),
    ("TSLA", "Tesla"), ("BRK.B", "Berkshire Hathaway Class B"), ("AVGO", "Broadcom"),
    ("JPM", "JPMorgan Chase"), ("LLY", "Eli Lilly"), ("V", "Visa"), ("MA", "Mastercard"),
    ("UNH", "UnitedHealth Group"), ("XOM", "Exxon Mobil"), ("JNJ", "Johnson & Johnson"), ("WMT", "Walmart"),
    ("PG", "Procter & Gamble"), ("HD", "Home Depot"), ("COST", "Costco"), ("NFLX", "Netflix"),
    ("BAC", "Bank of America"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"), ("ORCL", "Oracle"),
    ("AMD", "Advanced Micro Devices"), ("CRM", "Salesforce"), ("ADBE", "Adobe"), ("INTC", "Intel"),
    ("DIS", "Walt Disney"), ("CSCO", "Cisco Systems"), ("MCD", "McDonald's"), ("NKE", "Nike"),
    ("PFE", "Pfizer"), ("ABBV", "AbbVie"), ("MRK", "Merck"), ("CVX", "Chevron"), ("WFC", "Wells Fargo"),
    ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley"), ("BA", "Boeing"), ("CAT", "Caterpillar"),
    ("IBM", "IBM"), ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"), ("T", "AT&T"), ("VZ", "Verizon"),
    ("PYPL", "PayPal"), ("UBER", "Uber Technologies"), ("PLTR", "Palantir Technologies"), ("SBUX", "Starbucks"),
    ("SPY", "SPDR S&P 500 ETF"), ("VOO", "Vanguard S&P 500 ETF"), ("IVV", "iShares Core S&P 500 ETF"),
    ("QQQ", "Invesco QQQ (Nasdaq-100) ETF"), ("VTI", "Vanguard Total Stock Market ETF"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF"), ("IWM", "iShares Russell 2000 ETF"),
    ("VEA", "Vanguard Developed Markets ETF"), ("VWO", "Vanguard Emerging Markets ETF"),
    ("AGG", "iShares Core US Aggregate Bond ETF"), ("BND", "Vanguard Total Bond Market ETF"),
    ("GLD", "SPDR Gold Shares"), ("SCHD", "Schwab US Dividend Equity ETF"), ("VIG", "Vanguard Dividend Appreciation ETF"),
    ("XLK", "Technology Select Sector SPDR"), ("XLF", "Financial Select Sector SPDR"), ("XLE", "Energy Select Sector SPDR"),
)

_KEEP_UPPER = {
    "ETF", "ETN", "SPDR", "S&P", "QQQ", "ADR", "REIT", "US", "USA", "AI", "II", "III", "IV", "LP", "NV",
    "SA", "AG", "SE", "AB", "NA", "UK", "PLC", "IBM", "AT&T", "MSCI", "ESG", "TIPS", "CL", "JP", "EM", "USD",
}
_SPECIAL_CASE = {"ISHARES": "iShares", "PROSHARES": "ProShares", "WISDOMTREE": "WisdomTree", "JPMORGAN": "JPMorgan"}
_CLASS = re.compile(r"[-\s]+CL(?:ASS)?\s+([A-Z])\b(?:\s+SHARES)?")
_ADR = re.compile(r"[-\s]+(?:SP(?:ON)?\s+)?ADR\b")


@dataclass(frozen=True)
class SymbolEntry:
    """One searchable security."""

    symbol: str
    name: str
    kind: str  # "Stock", "ETF", "ADR", "REIT"

    @property
    def label(self) -> str:
        """What the search box shows: 'AAPL · Apple Inc.'."""
        return f"{self.symbol}{LABEL_SEPARATOR}{self.name}"


def clean_name(raw: str) -> str:
    """Tidy Finnhub's all-caps names for display."""
    name = raw.strip()
    name = re.sub(r"/THE$", "", name)  # 'COCA-COLA CO/THE'
    name = re.sub(r"-US$", "", name)  # 'SS SPDR S&P 500 ETF TRUST-US'
    name = re.sub(r"^SS SPDR\b", "SPDR", name)
    name = _CLASS.sub(lambda m: f" Class {m.group(1)}", name)
    name = _ADR.sub(" (ADR)", name)
    words = []
    for word in name.split():
        if word in _SPECIAL_CASE:
            words.append(_SPECIAL_CASE[word])
        elif word.strip("()") in _KEEP_UPPER or any(ch.isdigit() for ch in word) or not word.isupper():
            words.append(word)
        else:
            words.append("-".join(part.capitalize() for part in word.split("-")))
    return " ".join(words)


_KINDS = {"Common Stock": "Stock", "ETP": "ETF", "ADR": "ADR", "REIT": "REIT"}


def build_universe(rows: list[dict[str, Any]]) -> list[SymbolEntry]:
    """Popular names first (curated), then every other eligible symbol alphabetically."""
    eligible = {
        r["symbol"]: r
        for r in rows
        if r.get("symbol") and r.get("type") in SEARCHABLE_TYPES and r.get("mic") in MAJOR_EXCHANGES
    }
    popular = [
        SymbolEntry(sym, name, _KINDS.get(eligible[sym]["type"], "Stock") if sym in eligible else "Stock")
        for sym, name in POPULAR
    ]
    seen = {e.symbol for e in popular}
    rest = sorted(
        (
            SymbolEntry(sym, clean_name(r.get("description") or sym), _KINDS[r["type"]])
            for sym, r in eligible.items()
            if sym not in seen
        ),
        key=lambda e: e.symbol,
    )
    return popular + rest


def resolve_query(text: str, universe: list[SymbolEntry]) -> str | None:
    """Turn whatever was entered into a ticker.

    Accepts a picked label ('AAPL · Apple Inc.'), an exact ticker in any case
    ('brk.b'), or a company name fragment ('apple', which matches in list
    order, so popular companies win). Unknown input is returned as-is (upper-
    cased) so brand-new tickers can still be tried.
    """
    text = (text or "").strip()
    if not text:
        return None
    if LABEL_SEPARATOR in text:
        return text.split(LABEL_SEPARATOR, 1)[0].strip()
    upper = text.upper().lstrip("$")
    by_symbol = {e.symbol: e for e in universe}
    for candidate in (upper, upper.replace("-", "."), upper.replace(".", "-")):
        if candidate in by_symbol:
            return candidate
    needle = text.lower()
    squashed = needle.replace(" ", "")  # 'jp morgan' should still find 'JPMorgan'
    for entry in universe:
        name = entry.name.lower()
        if needle in name or squashed in name.replace(" ", ""):
            return entry.symbol
    return upper


def fallback_universe() -> list[SymbolEntry]:
    """The curated list alone, for when the full symbol list can't be fetched."""
    return build_universe([])


def fetch_universe(client: FinnhubClient | None = None) -> list[SymbolEntry]:
    """Download and build the full search universe (~11,000 entries)."""
    rows = (client or FinnhubClient(timeout=30)).get("/stock/symbol", exchange="US")
    return build_universe(rows if isinstance(rows, list) else [])
