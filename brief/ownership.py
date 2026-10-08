"""Ownership & insider activity section (stocks only).

Sources, and why two:
- Finnhub insider transactions carry the SEC transaction code (P = open-market
  purchase, S = open-market sale, A = grant, M = option exercise, F = tax
  withholding, G = gift, ...) and the shares held after each trade.
- yfinance carries each insider's role ("Chief Executive Officer"), which
  Finnhub lacks. yfinance's own descriptions are blank for many rows, so they
  can't classify trades, and its "insider purchases" summary counts grants and
  exercises as purchases (AAPL showed 12 "purchases" with zero real buys).

Only open-market trades (P and S) are treated as signals; everything else is
routine compensation activity and is summarized separately.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from brief.finnhub_client import FinnhubClient
from brief.market_data import first_number
from brief.models import DataUnavailableError, SectionResult, safe_section

logger = logging.getLogger(__name__)

SOURCE = "Finnhub (SEC Form 4 transactions), Yahoo Finance (roles, holders)"
LOOKBACK_DAYS = 183  # ~6 months

# Flag thresholds (also stated on the page).
CLUSTER_MIN_BUYERS = 3
CLUSTER_WINDOW_DAYS = 30
LARGE_SALE_VALUE = 5_000_000
LARGE_SALE_STAKE = 0.20
LARGE_SALE_STAKE_MIN_VALUE = 1_000_000  # ignore big-percentage but small-dollar sales

ROUTINE_CODES = {
    "A": "grants/awards",
    "M": "option exercises",
    "X": "option exercises",
    "F": "tax withholding",
    "G": "gifts",
    "D": "dispositions to the company",
    "C": "conversions",
}
_SENIOR_ROLE = re.compile(r"chief|\bceo\b|\bcfo\b|\bcoo\b|president|chair", re.IGNORECASE)


# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class InsiderTrade:
    """One insider's open-market trades on one day, combined."""

    name: str
    role: str | None
    side: str  # "buy" or "sell"
    shares: int
    value: float | None
    trade_date: date
    stake_change: float | None  # fraction of their holdings bought or sold that day

    @property
    def is_senior(self) -> bool:
        """C-suite, president, or chair."""
        return bool(self.role and _SENIOR_ROLE.search(self.role))


@dataclass(frozen=True)
class Flag:
    """A notable pattern worth calling out, with the rule that produced it."""

    kind: str  # "cluster_buying" | "executive_purchase" | "large_sale"
    tone: str  # "positive" or "caution"
    title: str
    detail: str


@dataclass(frozen=True)
class InsiderActivity:
    """Six months of insider trading: open-market trades, net flow, routine counts, flags."""

    trades: list[InsiderTrade]  # newest first
    routine_counts: dict[str, int]
    flags: list[Flag]
    start: date
    end: date

    def _total(self, side: str, attr: str) -> float:
        return sum(getattr(t, attr) or 0 for t in self.trades if t.side == side)

    @property
    def shares_bought(self) -> int:
        return int(self._total("buy", "shares"))

    @property
    def shares_sold(self) -> int:
        return int(self._total("sell", "shares"))

    @property
    def value_bought(self) -> float:
        return self._total("buy", "value")

    @property
    def value_sold(self) -> float:
        return self._total("sell", "value")

    @property
    def net_value(self) -> float:
        """Dollars bought minus dollars sold (negative = net selling)."""
        return self.value_bought - self.value_sold

    def people(self, side: str) -> int:
        """Distinct insiders who traded on this side."""
        return len({t.name for t in self.trades if t.side == side})

    def summary(self) -> str:
        """'3 insiders sold $86.8M; no open-market purchases.'"""
        sellers, buyers = self.people("sell"), self.people("buy")
        if not self.trades:
            return "No open-market insider purchases or sales in the past 6 months."
        parts = []
        if buyers:
            parts.append(f"{buyers} insider{'s' if buyers != 1 else ''} bought ${self.value_bought / 1e6:,.1f}M")
        if sellers:
            parts.append(f"{sellers} insider{'s' if sellers != 1 else ''} sold ${self.value_sold / 1e6:,.1f}M")
        if not buyers:
            parts.append("no open-market purchases")
        if not sellers:
            parts.append("no open-market sales")
        return "; ".join(parts) + " in the past 6 months."


@dataclass(frozen=True)
class InstitutionalHolder:
    """One of the largest institutional holders, from its latest 13F."""

    name: str
    pct_held: float | None
    shares: float | None
    value: float | None
    pct_change: float | None  # change in shares vs. the prior filing
    date_reported: date | None


@dataclass(frozen=True)
class OwnershipBreakdown:
    """Who owns the shares."""

    insider_pct: float | None
    institution_pct: float | None
    institution_count: int | None
    top_holders: list[InstitutionalHolder]


@dataclass(frozen=True)
class OwnershipActivity:
    """The section: either part can be missing if its source failed."""

    insiders: InsiderActivity | None
    ownership: OwnershipBreakdown | None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- roles


def _name_tokens(name: str) -> frozenset[str]:
    """'O'Brien Deirdre' and "O'BRIEN DEIRDRE" -> {'OBRIEN', 'DEIRDRE'}."""
    return frozenset(re.sub(r"[^A-Z ]", "", name.upper().replace("-", " ")).split())


def build_role_lookup(*frames: pd.DataFrame | None) -> dict[frozenset[str], str]:
    """Name tokens -> role, from yfinance tables with 'Insider'/'Name' and 'Position' columns."""
    roles: dict[frozenset[str], str] = {}
    for df in frames:
        if df is None or df.empty or "Position" not in df:
            continue
        name_col = "Insider" if "Insider" in df else "Name"
        for name, role in zip(df[name_col], df["Position"]):
            if isinstance(name, str) and isinstance(role, str) and role.strip():
                roles.setdefault(_name_tokens(name), role.strip())
    return roles


def find_role(name: str, roles: dict[frozenset[str], str]) -> str | None:
    """Match despite case, punctuation, word order, and a missing middle initial."""
    tokens = _name_tokens(name)
    if tokens in roles:
        return roles[tokens]
    for key, role in roles.items():
        if len(tokens & key) >= 2 and (tokens <= key or key <= tokens):
            return role
    return None


# ---------------------------------------------------------------- insiders


def display_name(raw: str) -> str:
    """Form 4 names are 'LAST FIRST M' in mixed case; show 'Last, First M'."""
    words = [w.title() if w.isupper() or w.islower() else w for w in raw.split()]
    return f"{words[0]}, {' '.join(words[1:])}" if len(words) > 1 else (words[0] if words else raw)


def parse_insider_activity(
    rows: list[dict[str, Any]], roles: dict[frozenset[str], str], end: date, lookback_days: int = LOOKBACK_DAYS
) -> InsiderActivity:
    """Combine Finnhub Form 4 rows into per-person-per-day open-market trades, plus routine counts and flags."""
    start = end - timedelta(days=lookback_days)
    grouped: dict[tuple[str, date, str], list[dict[str, Any]]] = defaultdict(list)
    routine: dict[str, int] = defaultdict(int)

    for row in rows:
        try:
            when = date.fromisoformat(row["transactionDate"])
        except (KeyError, TypeError, ValueError):
            continue
        if not start <= when <= end:
            continue
        code = row.get("transactionCode") or ""
        if code in ("P", "S") and not row.get("isDerivative"):
            grouped[(row.get("name") or "Unknown", when, "buy" if code == "P" else "sell")].append(row)
        elif code in ("P", "S"):
            routine["derivative trades"] += 1  # options etc., not shares bought or sold outright
        else:
            routine[ROUTINE_CODES.get(code, "other filings")] += 1

    trades = []
    for (name, when, side), group in grouped.items():
        shares = sum(abs(first_number(r, "change") or 0) for r in group)
        priced = [(abs(first_number(r, "change") or 0), first_number(r, "transactionPrice")) for r in group]
        value = sum(s * p for s, p in priced if p) or None
        held_after = [first_number(r, "share") for r in group if first_number(r, "share") is not None]
        stake = None
        if held_after and shares:
            # 'share' is holdings after each slice. Before a day of selling = lowest after + total sold;
            # before a day of buying = highest after - total bought.
            before = min(held_after) + shares if side == "sell" else max(held_after) - shares
            stake = shares / before if before > 0 else None
        trades.append(InsiderTrade(display_name(name), find_role(name, roles), side, int(shares), value, when, stake))

    trades.sort(key=lambda t: (t.trade_date, t.value or 0), reverse=True)
    return InsiderActivity(trades, dict(routine), detect_flags(trades), start, end)


def _fmt_date(d: date) -> str:
    """'Oct 2, 2026'."""
    return f"{d:%b} {d.day}, {d.year}"


def is_large_sale(t: InsiderTrade) -> bool:
    """$5M+ in a day, or 20%+ of the person's holdings when that's at least $1M."""
    value = t.value or 0
    return value >= LARGE_SALE_VALUE or (
        (t.stake_change or 0) >= LARGE_SALE_STAKE and value >= LARGE_SALE_STAKE_MIN_VALUE
    )


def detect_flags(trades: list[InsiderTrade]) -> list[Flag]:
    """Rule-based notable activity. Buys are stronger signals than sells, so they're flagged on any size."""
    flags: list[Flag] = []
    buys = sorted((t for t in trades if t.side == "buy"), key=lambda t: t.trade_date)

    # Cluster: CLUSTER_MIN_BUYERS different insiders buying within CLUSTER_WINDOW_DAYS.
    best: set[str] = set()
    for i, first in enumerate(buys):
        window = {t.name for t in buys[i:] if (t.trade_date - first.trade_date).days <= CLUSTER_WINDOW_DAYS}
        if len(window) > len(best):
            best = window
    if len(best) >= CLUSTER_MIN_BUYERS:
        flags.append(Flag(
            "cluster_buying", "positive", "Clustered insider buying",
            f"{len(best)} different insiders bought shares on the open market within {CLUSTER_WINDOW_DAYS} days. "
            "Several insiders buying at once is historically one of the more informative insider signals.",
        ))

    for t in buys:
        if t.is_senior:
            flags.append(Flag(
                "executive_purchase", "positive", f"Executive purchase: {t.name}",
                f"{t.role} bought {t.shares:,} shares"
                + (f" (${t.value / 1e6:,.1f}M)" if t.value else "")
                + f" on {_fmt_date(t.trade_date)}. Open-market buys by senior "
                "executives are uncommon and are made with their own money.",
            ))

    # Large sales: one flag per person, covering all of their qualifying sales in the window.
    # Applies to every insider: yfinance often labels senior executives just "Officer".
    by_person: dict[str, list[InsiderTrade]] = defaultdict(list)
    for t in trades:
        if t.side == "sell" and is_large_sale(t):
            by_person[t.name].append(t)
    for name, sales in by_person.items():
        sales.sort(key=lambda t: t.trade_date)
        total_value = sum(t.value or 0 for t in sales)
        total_shares = sum(t.shares for t in sales)
        biggest_stake = max((t.stake_change or 0) for t in sales)
        first, last = sales[0].trade_date, sales[-1].trade_date
        when = (
            f"on {_fmt_date(first)}" if first == last
            else f"across {len(sales)} days between {_fmt_date(first)} and {_fmt_date(last)}"
        )
        stake_note = f", up to {biggest_stake:.0%} of their holdings in a day" if biggest_stake else ""
        role = sales[0].role or "Insider"
        flags.append(Flag(
            "large_sale", "caution", f"Large sale: {name}",
            f"{role} sold {total_shares:,} shares (${total_value / 1e6:,.1f}M{stake_note}) {when}. "
            "Executive sales are often pre-scheduled (10b5-1 plans) or for diversification and taxes, "
            "so they are weaker signals than purchases.",
        ))
    return flags


# ---------------------------------------------------------------- institutions


def parse_ownership(major: pd.Series | dict | None, institutions: pd.DataFrame | None) -> OwnershipBreakdown:
    """yfinance major_holders values + institutional_holders table -> OwnershipBreakdown."""
    values = dict(major) if major is not None else {}
    count = first_number(values, "institutionsCount")
    holders = []
    if institutions is not None and not institutions.empty:
        for _, r in institutions.iterrows():
            reported = r.get("Date Reported")
            holders.append(InstitutionalHolder(
                name=str(r.get("Holder")),
                pct_held=first_number(r, "pctHeld"),
                shares=first_number(r, "Shares"),
                value=first_number(r, "Value"),
                pct_change=first_number(r, "pctChange"),
                date_reported=pd.Timestamp(reported).date() if pd.notna(reported) else None,
            ))
    return OwnershipBreakdown(
        insider_pct=first_number(values, "insidersPercentHeld"),
        institution_pct=first_number(values, "institutionsPercentHeld"),
        institution_count=int(count) if count is not None else None,
        top_holders=holders,
    )


# ---------------------------------------------------------------- fetching / section


def fetch_insider_inputs(symbol: str, start: date, client: FinnhubClient) -> tuple[list[dict], dict]:
    """Finnhub Form 4 rows and a yfinance role lookup."""
    data = client.get("/stock/insider-transactions", symbol=symbol, **{"from": start.isoformat()})
    rows = data.get("data", []) if isinstance(data, dict) else []
    ticker = yf.Ticker(symbol)
    try:
        roles = build_role_lookup(ticker.insider_transactions, ticker.insider_roster_holders)
    except Exception as exc:  # roles are optional: trades still show with a blank role
        logger.warning("Insider roles unavailable for %s: %s", symbol, exc)
        roles = {}
    return rows, roles


def fetch_ownership_inputs(symbol: str) -> tuple[pd.Series | None, pd.DataFrame | None]:
    """yfinance major_holders values and institutional_holders table."""
    ticker = yf.Ticker(symbol)
    major = ticker.major_holders
    return (major["Value"] if major is not None and "Value" in major else None), ticker.institutional_holders


@safe_section(SOURCE)
def get_ownership_activity(
    symbol: str, client: FinnhubClient | None = None, today: date | None = None
) -> SectionResult[OwnershipActivity]:
    """Ownership & insider section. Each half survives the other's failure."""
    today = today or date.today()
    insiders = ownership = None
    notes: list[str] = []
    try:
        rows, roles = fetch_insider_inputs(symbol, today - timedelta(days=LOOKBACK_DAYS), client or FinnhubClient())
        insiders = parse_insider_activity(rows, roles, today)
    except DataUnavailableError as exc:
        notes.append(f"Insider transactions unavailable: {exc}")
    try:
        ownership = parse_ownership(*fetch_ownership_inputs(symbol))
    except Exception as exc:  # yfinance raises assorted errors for uncovered tickers
        logger.warning("Ownership data unavailable for %s: %s", symbol, exc)
        notes.append("Institutional ownership data unavailable.")
    if insiders is None and (ownership is None or not ownership.top_holders):
        raise DataUnavailableError("No ownership or insider data available.")
    return SectionResult.success(OwnershipActivity(insiders, ownership, notes), SOURCE)
