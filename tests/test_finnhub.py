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


# ---------------------------------------------------------------- earnings track record


@pytest.mark.parametrize(
    "actual, estimate, expected",
    [(2.84, 2.7257, (2.84 - 2.7257) / 2.7257), (-0.5, -1.0, 0.5), (1.0, 0, None), (None, 1.0, None)],
)
def test_surprise_fraction(actual, estimate, expected) -> None:
    assert fh.surprise_fraction(actual, estimate) == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("surprise, outcome", [(0.042, "beat"), (0.01, "beat"), (-0.0089, "in line"), (-0.01, "miss"), (None, None)])
def test_outcome_band(surprise, outcome) -> None:
    q = fh.EarningsResult(date(2026, 6, 30), 3, 2026, 1.0, 1.0, surprise)
    assert q.outcome == outcome


def test_earnings_history_from_captured_aapl() -> None:
    h = fh.parse_earnings_history(load_json("finnhub_earnings_history_aapl.json"))
    assert len(h.quarters) == 4
    assert h.quarters[0].period_end == date(2026, 6, 30)  # newest first
    assert h.quarters[0].outcome == "in line"  # -0.9% is within the ±1% band
    assert h.count("beat") == 3
    assert h.summary().startswith("Beat estimates in 3 of the last 4 quarters (1 in line), average surprise +")


def test_earnings_summary_with_miss() -> None:
    qs = [fh.EarningsResult(date(2026, 3, 31), 1, 2026, 1.0, 0.9, -0.1), fh.EarningsResult(date(2025, 12, 31), 4, 2025, 1.0, 1.1, 0.1)]
    assert fh.EarningsHistory(qs).summary() == "Beat estimates in 1 of the last 2 quarters (1 missed), average surprise +0.0%."


def test_earnings_history_empty_for_etf() -> None:
    result = fh.get_earnings_history("SPY", client=client(FakeResponse(200, [])))
    assert not result.ok and result.error == "No earnings history available."


def test_earnings_summary_all_beats() -> None:
    qs = [fh.EarningsResult(date(2026, m, 28), 1, 2026, 1.0, 1.05, 0.05) for m in (3, 6)]
    assert fh.EarningsHistory(qs).summary() == "Beat estimates in all of the last 2 quarters, average surprise +5.0%."
