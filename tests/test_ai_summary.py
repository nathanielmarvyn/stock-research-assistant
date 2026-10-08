"""Tests for the at-a-glance AI summary: facts, retry on invented figures, and failure handling."""

from types import SimpleNamespace

import anthropic
import httpx2
import pydantic
import pytest

from brief import ai_summary as s
from brief.financials import parse_financials
from brief.market_data import AssetType, TickerInfo, parse_snapshot
from brief.models import SectionResult

AAPL = TickerInfo("AAPL", "Apple Inc.", AssetType.STOCK, "NMS", "USD")


def ok(data) -> SectionResult:
    return SectionResult.success(data, "test")


@pytest.fixture
def sections(aapl_info, aapl_statements) -> dict:
    return {"snapshot": ok(parse_snapshot(aapl_info)), "financials": ok(parse_financials(*aapl_statements, aapl_info))}


class FakeMessages:
    """Returns queued summaries (or raises queued exceptions) and records each request."""

    def __init__(self, *replies) -> None:
        self.replies, self.requests = list(replies), []

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        parsed = s.SummaryOutput(summary=reply) if reply is not None else None
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn")


def fake_llm(*replies) -> SimpleNamespace:
    return SimpleNamespace(messages=FakeMessages(*replies))


# ---------------------------------------------------------------- description


def test_trim_description_cuts_at_sentence() -> None:
    text = "Apple designs phones. " * 80
    trimmed = s.trim_description(text, limit=100)
    assert len(trimmed) <= 100 and trimmed.endswith(".")


def test_trim_description_handles_missing() -> None:
    assert s.trim_description(None) is None and s.trim_description("Short.") == "Short."


def test_facts_include_business_description(sections) -> None:
    facts = s.summary_facts(AAPL, "Apple designs, manufactures, and markets smartphones.", **sections)
    assert facts["business_description"].startswith("Apple designs")
    assert "financials" in facts


# ---------------------------------------------------------------- section


def test_grounded_summary_needs_one_call(sections) -> None:
    llm = fake_llm("Apple makes iPhones. Revenue grew +16.4% year over year. The biggest risk is valuation.")
    result = s.get_ai_summary(AAPL, "Apple designs phones.", llm=llm, **sections)
    assert result.ok and result.data.unverified_numbers == []
    assert len(llm.messages.requests) == 1
    assert llm.messages.requests[0]["output_format"] is s.SummaryOutput


def test_invented_figure_triggers_one_retry(sections) -> None:
    llm = fake_llm("Revenue grew 99.9% year over year.", "Revenue grew +16.4% year over year.")
    result = s.get_ai_summary(AAPL, "Apple designs phones.", llm=llm, **sections)
    assert result.ok and result.data.text == "Revenue grew +16.4% year over year."
    retry = llm.messages.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "99.9%" in retry[2]["content"]


def test_still_unverified_after_retry_is_flagged(sections) -> None:
    llm = fake_llm("Growth was 99.9%.", "Growth was 88.8%.")
    result = s.get_ai_summary(AAPL, None, llm=llm, **sections)
    assert result.ok and result.data.unverified_numbers == ["88.8%"]
    assert len(llm.messages.requests) == 2  # no endless retrying


def test_cut_off_reply_is_a_clean_failure(sections) -> None:
    try:
        s.SummaryOutput.model_validate_json('{"summary": "Apple ma')
    except pydantic.ValidationError as exc:
        truncated = exc
    result = s.get_ai_summary(AAPL, None, llm=fake_llm(truncated), **sections)
    assert not result.ok and "incomplete" in result.error


def test_api_error_is_friendly(sections) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError("slow", response=httpx2.Response(429, request=request), body=None)
    result = s.get_ai_summary(AAPL, None, llm=fake_llm(error), **sections)
    assert not result.ok and "rate-limiting" in result.error


def test_needs_core_data() -> None:
    result = s.get_ai_summary(AAPL, "Apple designs phones.", llm=fake_llm("x"))
    assert not result.ok and "Not enough data" in result.error
