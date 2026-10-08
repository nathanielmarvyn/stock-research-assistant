"""Application settings and secret loading.

Secrets are resolved in this order:
1. Environment variables (populated from a local ``.env`` file via python-dotenv)
2. Streamlit secrets (``.streamlit/secrets.toml`` or a host's secrets manager)

Nothing secret is ever hardcoded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str) -> str | None:
    """Return a secret from the environment or Streamlit secrets, or None if unset."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)  # raises if no secrets file exists
    except Exception:
        return None


@dataclass(frozen=True)
class Settings:
    """Tunable, non-secret configuration plus resolved API keys."""

    finnhub_api_key: str | None
    anthropic_api_key: str | None

    # Claude models: a cheap one for repetitive headline scoring,
    # a stronger one for the written analysis.
    sentiment_model: str = "claude-haiku-5-5"
    analysis_model: str = "claude-sonnet-5-5"
    # Labeling headlines is simple: low effort matched higher settings' labels
    # in testing while running fastest. Thinking tokens count toward max_tokens.
    sentiment_effort: str = "low"
    sentiment_max_tokens: int = 3000
    # The at-a-glance summary is a separate fast call so it doesn't wait on the full analysis.
    summary_model: str = "claude-haiku-5-5"
    # Medium effort: low was faster (~2.5s vs ~6s) but sometimes garbled or ran long, and
    # this is the most-read block. Thinking counts toward max_tokens, so leave headroom.
    summary_effort: str = "medium"
    summary_max_tokens: int = 6000
    # Sonnet's adaptive thinking counts toward max_tokens, so leave headroom.
    # Worst case at $10/M output tokens is about $0.08 per analysis.
    analysis_max_tokens: int = 8000
    analysis_effort: str = "medium"  # low | medium | high | xhigh | max
    # Cost guard for a public demo: distinct tickers per browser session that
    # get a (paid) AI analysis. Cached repeats don't count.
    max_ai_analyses_per_session: int = 10

    cache_ttl_seconds: int = 15 * 60
    request_timeout_seconds: int = 10
    benchmark_ticker: str = "SPY"
    news_lookback_days: int = 14
    max_headlines: int = 10


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings once per process."""
    return Settings(
        finnhub_api_key=get_secret("FINNHUB_API_KEY"),
        anthropic_api_key=get_secret("ANTHROPIC_API_KEY"),
    )
