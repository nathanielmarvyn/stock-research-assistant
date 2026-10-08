"""Finnhub client (free tier): earnings calendar, analyst recommendations, company news.

``FinnhubClient`` handles HTTP concerns (auth, timeouts, 429 retries) and
returns raw JSON. Pure ``parse_*`` functions turn that JSON into typed data and
are tested offline against captured responses.

Price targets are a paid Finnhub feature, so the Wall Street view takes them
from yfinance's info payload instead (and labels the source).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests

from brief.config import get_settings
from brief.market_data import SOURCE as YAHOO_SOURCE
from brief.market_data import DataSourceError, first_number
from brief.models import DataUnavailableError, SectionResult, safe_section

logger = logging.getLogger(__name__)

SOURCE = "Finnhub"
BASE_URL = "https://finnhub.io/api/v1"
MAX_RETRIES = 2
EARNINGS_LOOKAHEAD_DAYS = 120

# Finnhub's "hour" codes for earnings announcements.
_HOUR_LABELS = {"bmo": "before market open", "amc": "after market close", "dmh": "during market hours"}

# 1 = Strong Buy ... 5 = Strong Sell, the common sell-side scoring scale.
_RATING_WEIGHTS = {"strongBuy": 1, "buy": 2, "hold": 3, "sell": 4, "strongSell": 5}
_RATING_LABELS = ((1.5, "Strong Buy"), (2.5, "Buy"), (3.5, "Hold"), (4.5, "Sell"), (5.0, "Strong Sell"))


# ---------------------------------------------------------------- HTTP client


class FinnhubClient:
    """Minimal Finnhub REST client with retry on rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Create a client; defaults come from application settings."""
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.finnhub_api_key
        self.timeout = timeout or settings.request_timeout_seconds
        self.session = session or requests.Session()

    def get(self, path: str, **params: Any) -> Any:
        """GET a Finnhub endpoint and return parsed JSON.

        Raises DataUnavailableError for missing keys or forbidden endpoints and
        DataSourceError for rate limits, timeouts, and server errors.
        """
        if not self.api_key:
            raise DataUnavailableError("Finnhub API key is not configured.")

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    f"{BASE_URL}{path}",
                    params=params,
                    headers={"X-Finnhub-Token": self.api_key},
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                raise DataSourceError("Finnhub timed out. Try again shortly.") from exc
            except requests.RequestException as exc:
                raise DataSourceError(f"Could not reach Finnhub: {exc}") from exc

            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = 2**attempt  # 1s, 2s
                    logger.warning("Finnhub rate limit hit; retrying in %ss", wait)
                    time.sleep(wait)
                    continue
                raise DataSourceError("Finnhub rate limit reached (60 calls/min). Try again in a minute.")
            if response.status_code == 401:
                raise DataUnavailableError("Finnhub rejected the API key.")
            if response.status_code == 403:
                raise DataUnavailableError("This Finnhub endpoint isn't available on the free plan.")
            if not response.ok:
                raise DataSourceError(f"Finnhub error {response.status_code}.")
            return response.json()
        raise AssertionError("unreachable")

    def earnings_calendar(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        """Scheduled earnings releases for ``symbol`` between two dates."""
        data = self.get("/calendar/earnings", symbol=symbol, **{"from": start.isoformat(), "to": end.isoformat()})
        return data.get("earningsCalendar", []) if isinstance(data, dict) else []

    def recommendation_trends(self, symbol: str) -> list[dict[str, Any]]:
        """Monthly analyst rating counts, newest first."""
        data = self.get("/stock/recommendation", symbol=symbol)
        return data if isinstance(data, list) else []

    def company_news(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        """Company news articles published between two dates."""
        data = self.get("/company-news", symbol=symbol, **{"from": start.isoformat(), "to": end.isoformat()})
        return data if isinstance(data, list) else []


# ---------------------------------------------------------------- earnings


@dataclass(frozen=True)
class EarningsEvent:
    """The next scheduled earnings release."""

    date: date
    timing: str | None  # e.g. "after market close"
    fiscal_quarter: int | None
    fiscal_year: int | None
    eps_estimate: float | None
    revenue_estimate: float | None


def parse_next_earnings(calendar: list[dict[str, Any]], today: date) -> EarningsEvent | None:
    """Pick the soonest release on or after ``today`` (Finnhub doesn't sort the list)."""
    upcoming = []
    for item in calendar:
        try:
            when = date.fromisoformat(item["date"])
        except (KeyError, TypeError, ValueError):
            continue
        if when >= today:
            upcoming.append((when, item))
    if not upcoming:
        return None
    when, item = min(upcoming, key=lambda pair: pair[0])
    return EarningsEvent(
        date=when,
        timing=_HOUR_LABELS.get(item.get("hour") or ""),
        fiscal_quarter=item.get("quarter"),
        fiscal_year=item.get("year"),
        eps_estimate=first_number(item, "epsEstimate"),
        revenue_estimate=first_number(item, "revenueEstimate"),
    )


@safe_section(SOURCE)
def get_next_earnings(
    symbol: str, client: FinnhubClient | None = None, today: date | None = None
) -> SectionResult[EarningsEvent]:
    """Next earnings date. Fails softly when none is scheduled (e.g. ETFs)."""
    client = client or FinnhubClient()
    today = today or date.today()
    calendar = client.earnings_calendar(symbol, today, today + timedelta(days=EARNINGS_LOOKAHEAD_DAYS))
    event = parse_next_earnings(calendar, today)
    if event is None:
        raise DataUnavailableError("No upcoming earnings date announced.")
    return SectionResult.success(event, SOURCE)


# ---------------------------------------------------------------- Wall Street view


@dataclass(frozen=True)
class AnalystConsensus:
    """Analyst rating distribution for one month."""

    period: date
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    score: float  # 1 (Strong Buy) .. 5 (Strong Sell)
    label: str
    prior_score: float | None  # previous month, to show drift

    @property
    def total(self) -> int:
        """Number of analysts with a rating."""
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell


@dataclass(frozen=True)
class PriceTarget:
    """Average analyst 12-month price target and implied move."""

    mean: float | None
    high: float | None
    low: float | None
    analyst_count: int | None
    upside: float | None  # fraction vs. current price


@dataclass(frozen=True)
class WallStreetView:
    """Wall Street section: consensus (Finnhub) plus price target (Yahoo)."""

    consensus: AnalystConsensus | None
    price_target: PriceTarget | None


def rating_score(counts: dict[str, Any]) -> float | None:
    """Analyst-weighted score on the 1 (Strong Buy) to 5 (Strong Sell) scale."""
    total = sum(int(counts.get(k) or 0) for k in _RATING_WEIGHTS)
    if total == 0:
        return None
    return sum(w * int(counts.get(k) or 0) for k, w in _RATING_WEIGHTS.items()) / total


def rating_label(score: float) -> str:
    """Map a 1–5 score to its consensus label."""
    return next(label for limit, label in _RATING_LABELS if score <= limit)


def parse_consensus(trends: list[dict[str, Any]]) -> AnalystConsensus | None:
    """Latest month's consensus, with the prior month's score for comparison."""
    months = sorted(
        (t for t in trends if rating_score(t) is not None),
        key=lambda t: t.get("period", ""),
        reverse=True,
    )
    if not months:
        return None
    latest = months[0]
    score = rating_score(latest)
    return AnalystConsensus(
        period=date.fromisoformat(latest["period"]),
        strong_buy=int(latest.get("strongBuy") or 0),
        buy=int(latest.get("buy") or 0),
        hold=int(latest.get("hold") or 0),
        sell=int(latest.get("sell") or 0),
        strong_sell=int(latest.get("strongSell") or 0),
        score=score,
        label=rating_label(score),
        prior_score=rating_score(months[1]) if len(months) > 1 else None,
    )


def parse_price_target(info: dict[str, Any]) -> PriceTarget | None:
    """Price target fields from yfinance's info dict; None when not covered."""
    mean = first_number(info, "targetMeanPrice")
    if mean is None:
        return None
    price = first_number(info, "currentPrice", "regularMarketPrice")
    count = first_number(info, "numberOfAnalystOpinions")
    return PriceTarget(
        mean=mean,
        high=first_number(info, "targetHighPrice"),
        low=first_number(info, "targetLowPrice"),
        analyst_count=int(count) if count is not None else None,
        upside=(mean / price - 1) if price else None,
    )


WALL_STREET_SOURCE = f"{SOURCE} (ratings), {YAHOO_SOURCE} (price targets)"


@safe_section(WALL_STREET_SOURCE)
def get_wall_street_view(
    symbol: str, info: dict[str, Any], client: FinnhubClient | None = None
) -> SectionResult[WallStreetView]:
    """Wall Street section. Either half can be missing; fails only if both are."""
    consensus, consensus_error = None, None
    try:
        consensus = parse_consensus((client or FinnhubClient()).recommendation_trends(symbol))
    except DataUnavailableError as exc:  # keep going: the price target may still exist
        consensus_error = str(exc)
        logger.warning("Analyst ratings unavailable for %s: %s", symbol, exc)

    target = parse_price_target(info)
    if consensus is None and target is None:
        raise DataUnavailableError(consensus_error or "No analyst coverage available.")
    return SectionResult.success(WallStreetView(consensus, target), WALL_STREET_SOURCE)
