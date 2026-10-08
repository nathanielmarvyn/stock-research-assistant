"""Tests for brief.finnhub_client parsing and HTTP error handling (no network)."""

from datetime import date

import pytest
import requests

from brief import finnhub_client as fh
from brief.market_data import DataSourceError
from brief.models import DataUnavailableError
from tests.conftest import load_json


# ---------------------------------------------------------------- earnings


def test_next_earnings_picks_soonest_upcoming() -> None:
    calendar = load_json("finnhub_earnings_aapl.json")["earningsCalendar"]
    # The captured list is unsorted (Jan 2027 listed before Oct 2026).
    event = fh.parse_next_earnings(calendar, today=date(2026, 10, 8))
    assert event.date == date(2026, 10, 29)
    assert event.timing == "after market close"
    assert event.eps_estimate is not None


def test_next_earnings_skips_past_and_bad_rows() -> None:
    calendar = [
        {"date": "2026-01-01", "hour": "bmo"},
        {"date": "not-a-date"},
        {"date": "2026-11-05", "hour": "bmo", "quarter": 3, "year": 2026},
    ]
    event = fh.parse_next_earnings(calendar, today=date(2026, 10, 8))
    assert event.date == date(2026, 11, 5) and event.timing == "before market open"


def test_next_earnings_none_when_nothing_scheduled() -> None:
    assert fh.parse_next_earnings([], today=date(2026, 10, 8)) is None


# ---------------------------------------------------------------- consensus


def test_rating_score_hand_calculated() -> None:
    # 2 strong buy (1), 1 hold (3), 1 strong sell (5) -> (2 + 3 + 5) / 4 = 2.5
    assert fh.rating_score({"strongBuy": 2, "hold": 1, "strongSell": 1}) == 2.5
    assert fh.rating_score({}) is None


@pytest.mark.parametrize(
    "score, label",
    [(1.0, "Strong Buy"), (1.5, "Strong Buy"), (2.23, "Buy"), (3.0, "Hold"), (4.0, "Sell"), (4.9, "Strong Sell")],
)
def test_rating_label(score: float, label: str) -> None:
    assert fh.rating_label(score) == label


def test_consensus_from_captured_aapl() -> None:
    trends = load_json("finnhub_recommendations_aapl.json")
    c = fh.parse_consensus(trends)
    latest = max(trends, key=lambda t: t["period"])
    assert c.period == date.fromisoformat(latest["period"])
    assert c.total == sum(latest[k] for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
    assert 1 <= c.score <= 5 and c.prior_score is not None


def test_consensus_empty_for_etf() -> None:
    assert fh.parse_consensus([]) is None


def test_price_target_upside(aapl_info: dict) -> None:
    pt = fh.parse_price_target(aapl_info)
    assert pt.mean == aapl_info["targetMeanPrice"]
    assert pt.upside == pytest.approx(aapl_info["targetMeanPrice"] / aapl_info["currentPrice"] - 1)


def test_price_target_none_without_coverage(spy_info: dict) -> None:
    assert fh.parse_price_target(spy_info) is None


# ---------------------------------------------------------------- HTTP handling


class FakeResponse:
    def __init__(self, status: int, payload=None) -> None:
        self.status_code, self._payload = status, payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, *responses) -> None:
        self.responses, self.calls = list(responses), 0

    def get(self, *args, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch) -> None:
    monkeypatch.setattr(fh.time, "sleep", lambda _: None)


def client(*responses) -> fh.FinnhubClient:
    return fh.FinnhubClient(api_key="test", timeout=1, session=FakeSession(*responses))


def test_retries_after_rate_limit() -> None:
    c = client(FakeResponse(429), FakeResponse(200, [{"period": "2026-09-01", "buy": 1}]))
    assert c.recommendation_trends("AAPL") == [{"period": "2026-09-01", "buy": 1}]
    assert c.session.calls == 2


def test_gives_up_after_repeated_rate_limits() -> None:
    c = client(*[FakeResponse(429)] * (fh.MAX_RETRIES + 1))
    with pytest.raises(DataSourceError, match="rate limit"):
        c.recommendation_trends("AAPL")


@pytest.mark.parametrize(
    "status, error, message",
    [(401, DataUnavailableError, "rejected"), (403, DataUnavailableError, "free plan"), (500, DataSourceError, "500")],
)
def test_http_errors(status, error, message) -> None:
    with pytest.raises(error, match=message):
        client(FakeResponse(status)).recommendation_trends("AAPL")


def test_timeout_is_friendly() -> None:
    with pytest.raises(DataSourceError, match="timed out"):
        client(requests.Timeout()).recommendation_trends("AAPL")


def test_missing_key() -> None:
    with pytest.raises(DataUnavailableError, match="not configured"):
        fh.FinnhubClient(api_key="").get("/quote")


# ---------------------------------------------------------------- sections


def test_wall_street_survives_ratings_failure(aapl_info: dict) -> None:
    """Finnhub down -> still show the Yahoo price target."""
    result = fh.get_wall_street_view("AAPL", aapl_info, client=client(FakeResponse(401)))
    assert result.ok
    assert result.data.consensus is None and result.data.price_target is not None


def test_wall_street_fails_with_no_coverage(spy_info: dict) -> None:
    result = fh.get_wall_street_view("SPY", spy_info, client=client(FakeResponse(200, [])))
    assert not result.ok and "No analyst coverage" in result.error


def test_earnings_section_for_etf_fails_softly() -> None:
    result = fh.get_next_earnings("SPY", client=client(FakeResponse(200, {"earningsCalendar": []})))
    assert not result.ok and result.error == "No upcoming earnings date announced."
