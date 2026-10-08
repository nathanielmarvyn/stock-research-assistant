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
    sentiment_model: str = "claude-haiku-4-5"
    analysis_model: str = "claude-sonnet-5-5"
    sentiment_max_tokens: int = 1500
    analysis_max_tokens: int = 1500

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
