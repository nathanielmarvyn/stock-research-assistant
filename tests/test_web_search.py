"""Tests for the Perplexity Search wrapper: parsing, dedupe, queries, and errors (offline)."""

from datetime import date
from types import SimpleNamespace

import httpx
import perplexity
import pytest

from brief import web_search
from brief.models import DataUnavailableError

SINCE = date(2026, 10, 1)


def result(title: str, url: str, day: str | None = "2026-10-10", snippet: str = "Snippet.", updated: str | None = None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, date=day, last_updated=updated)


class FakeSearch:
    """Records the request and returns canned results."""

    def __init__(self, results=(), error: Exception | None = None) -> None:
        self.results, self.error, self.calls = list(results), error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(id="req-1", results=self.results)


def fake_client(results=(), error=None) -> SimpleNamespace:
    return SimpleNamespace(search=FakeSearch(results, error))


def test_to_articles_maps_fields_to_finnhub_shape() -> None:
    [art] = web_search.to_articles([result("Apple beats", "https://www.reuters.com/apple/", snippet=" Up 8%. ")], SINCE)
    assert art == {
        "headline": "Apple beats",
        "url": "https://www.reuters.com/apple/",
        "source": "reuters.com",
        "datetime": 1791590400,  # 2026-10-10 00:00 UTC
        "summary": "Up 8%.",
    }


def test_to_articles_dedupes_by_url_across_queries() -> None:
    results = [
        result("A", "https://www.cnbc.com/story/"),
        [result("A again", "https://cnbc.com/story"), result("B", "https://wsj.com/b")],  # per-query list
    ]
    assert [a["headline"] for a in web_search.to_articles(results, SINCE)] == ["A", "B"]


def test_to_articles_drops_undated_old_and_incomplete() -> None:
    results = [
        result("No date", "https://a.com/1", day=None),
        result("Old", "https://a.com/2", day="2026-09-20"),
        result("", "https://a.com/3"),
        result("Bad date", "https://a.com/4", day="last week"),
        result("Updated only", "https://a.com/5", day=None, updated="2026-10-12T08:00:00Z"),
        {"title": "Dict result", "url": "https://a.com/6", "snippet": "s", "date": "2026-10-02"},
    ]
    assert [a["headline"] for a in web_search.to_articles(results, SINCE)] == ["Updated only", "Dict result"]


def test_url_key_ignores_www_case_and_trailing_slash() -> None:
    assert web_search.url_key("https://WWW.Reuters.com/a/") == web_search.url_key("http://reuters.com/a")


def test_news_queries() -> None:
    assert web_search.news_queries("AAPL", "Apple Inc.", ("Apple",), is_etf=False) == ["Apple Inc. (AAPL) stock news"]
    assert web_search.news_queries("SPY", "SPDR S&P 500 ETF", ("S&P 500",), is_etf=True) == [
        "SPY ETF news", "S&P 500 index news",
    ]
    many = web_search.news_queries("X", "X", tuple(f"k{i}" for i in range(9)), is_etf=True)
    assert len(many) == web_search.MAX_QUERIES


def test_search_news_sends_documented_parameters() -> None:
    client = fake_client([result("Apple beats", "https://reuters.com/a")])
    articles = web_search.search_news(["Apple Inc. (AAPL) stock news"], SINCE, 10, client)
    assert [a["headline"] for a in articles] == ["Apple beats"]
    assert client.search.calls == [{
        "query": "Apple Inc. (AAPL) stock news",  # a single query is sent as a string
        "max_results": 10,
        "search_after_date_filter": "10/01/2026",  # MM/DD/YYYY per the API reference
        "search_language_filter": ["en"],
    }]
    web_search.search_news(["q1", "q2"], SINCE, 5, client)
    assert client.search.calls[-1]["query"] == ["q1", "q2"]


def test_search_news_with_no_queries_makes_no_request() -> None:
    client = fake_client()
    assert web_search.search_news([], SINCE, 10, client) == [] and client.search.calls == []


def test_client_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_search, "get_settings", lambda: SimpleNamespace(perplexity_api_key=None))
    with pytest.raises(DataUnavailableError, match="not configured"):
        web_search.perplexity_client()


@pytest.mark.parametrize(
    "status, cls, expected",
    [
        (401, perplexity.AuthenticationError, "key was rejected"),
        (429, perplexity.RateLimitError, "rate-limiting"),
        (500, perplexity.InternalServerError, "returned an error (500)"),
    ],
)
def test_describe_search_error(status: int, cls: type, expected: str) -> None:
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.perplexity.ai/search"))
    assert expected in web_search.describe_search_error(cls("x", response=response, body=None))
