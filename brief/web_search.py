"""Web search (Perplexity Search API): extra headlines when Finnhub's news is thin.

Finnhub's company-news feed is sparse for ETFs and smaller companies. When it
yields too few relevant headlines, the news section asks the Perplexity Search
API (``POST /search``) for recent web results and merges them in. The Search
API returns ranked raw results (title, url, snippet, date), not an LLM answer,
so they go through the same relevance filter and Claude sentiment scoring as
Finnhub's headlines.

Results are converted to Finnhub's article shape (``headline``, ``url``,
``source``, ``datetime``, ``summary``) so ``news.select_headlines`` handles both
sources. Undated results are dropped: the brief shows when each item was published.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlsplit

import perplexity
from perplexity import Perplexity

from brief.config import get_settings
from brief.models import DataUnavailableError

SOURCE = "Perplexity Search API (web)"
MAX_QUERIES = 5  # the API accepts up to five queries per request


class SearchClient(Protocol):
    """The slice of the Perplexity client used here (a fake in tests)."""

    @property
    def search(self) -> Any: ...


def perplexity_client() -> Perplexity:
    """Perplexity client from settings; raises DataUnavailableError without a key."""
    settings = get_settings()
    if not settings.perplexity_api_key:
        raise DataUnavailableError("Perplexity API key is not configured.")
    # The SDK retries 429s and 5xx itself, honoring Retry-After.
    return Perplexity(api_key=settings.perplexity_api_key, timeout=float(settings.request_timeout_seconds), max_retries=2)


def describe_search_error(exc: Exception) -> str:
    """Short description of a Perplexity API failure (never includes the key)."""
    if isinstance(exc, perplexity.AuthenticationError):
        return "the Perplexity API key was rejected"
    if isinstance(exc, perplexity.RateLimitError):
        return "the Perplexity API is rate-limiting requests"
    if isinstance(exc, perplexity.APIConnectionError):
        return "the Perplexity API could not be reached"
    if isinstance(exc, perplexity.APIStatusError):
        return f"the Perplexity API returned an error ({exc.status_code})"
    return str(exc)


def url_key(url: str) -> str:
    """URL normalized for duplicate detection: host without www, path without trailing slash."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}"


def site_name(url: str) -> str:
    """Display source for a web result ('https://www.reuters.com/x' -> 'reuters.com')."""
    return urlsplit(url).netloc.lower().removeprefix("www.") or "Web"


def parse_date(value: str | None) -> datetime | None:
    """A result's ``date``/``last_updated`` (YYYY-MM-DD, possibly with a time) as UTC midnight."""
    if not value:
        return None
    try:
        day = date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def _field(result: Any, name: str) -> Any:
    """Read a field from an SDK result object or a plain dict."""
    return result.get(name) if isinstance(result, dict) else getattr(result, name, None)


def to_articles(results: Iterable[Any], since: date) -> list[dict[str, Any]]:
    """Search results as Finnhub-style article dicts, deduped by URL, dated on/after ``since``."""
    seen: set[str] = set()
    articles: list[dict[str, Any]] = []
    # The SDK types multi-query results as one flat list; flatten per-query lists just in case.
    flat = (r for item in results for r in (item if isinstance(item, list) else [item]))
    for result in flat:
        title, url = (_field(result, "title") or "").strip(), (_field(result, "url") or "").strip()
        published = parse_date(_field(result, "date")) or parse_date(_field(result, "last_updated"))
        if not title or not url or published is None or published.date() < since:
            continue
        key = url_key(url)
        if key in seen:
            continue
        seen.add(key)
        articles.append({
            "headline": title,
            "url": url,
            "source": site_name(url),
            "datetime": int(published.timestamp()),
            "summary": (_field(result, "snippet") or "").strip(),
        })
    return articles


def news_queries(symbol: str, company_name: str, keywords: Sequence[str], is_etf: bool) -> list[str]:
    """One to five specific queries: the ticker with its company or index name, plus 'news'."""
    if is_etf:
        queries = [f"{symbol} ETF news", *(f"{k} index news" for k in keywords)]
    else:
        queries = [f"{company_name} ({symbol}) stock news"]
    return list(dict.fromkeys(queries))[:MAX_QUERIES]


def search_news(
    queries: Sequence[str],
    since: date,
    max_results: int,
    client: SearchClient | None = None,
) -> list[dict[str, Any]]:
    """Recent web results for ``queries`` as article dicts. Raises perplexity.APIError on failure."""
    if not queries:
        return []
    response = (client or perplexity_client()).search.create(
        query=list(queries) if len(queries) > 1 else queries[0],
        max_results=max_results,
        search_after_date_filter=since.strftime("%m/%d/%Y"),
        search_language_filter=["en"],
    )
    return to_articles(response.results, since)
