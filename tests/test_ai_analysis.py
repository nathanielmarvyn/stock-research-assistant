"""Tests for the AI analysis facts document, grounding check, and error handling."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from brief import ai_analysis as ai
from brief.financials import parse_financials
from brief.finnhub_client import WallStreetView, parse_price_target
from brief.market_data import AssetType, TickerInfo, parse_etf_profile, parse_snapshot, parse_top_holdings
from brief.models import SectionResult

AAPL = TickerInfo("AAPL", "Apple Inc.", AssetType.STOCK, "NMS", "USD")
SPY = TickerInfo("SPY", "SPDR S&P 500 ETF Trust", AssetType.ETF, "PCX", "USD")


def ok(data) -> SectionResult:
    return SectionResult.success(data, "test")


@pytest.fixture
def aapl_sections(aapl_info, aapl_statements) -> dict:
    return {
        "snapshot": ok(parse_snapshot(aapl_info)),
        "financials": ok(parse_financials(*aapl_statements, aapl_info)),
        "wall_street": ok(WallStreetView(None, parse_price_target(aapl_info))),
    }


def sample_analysis(**overrides) -> ai.Analysis:
    fields = {
        "summary": "Revenue grew +16.4% YoY to $109.42B.",
        "bull_case": ["Gross margin of 50.1% is high."],
        "bear_case": ["Trailing P/E of 38.3 is rich."],
        "key_risks": ["Concentration risk."],
        "what_to_watch": ["Next earnings."],
    }
    return ai.Analysis(**{**fields, **overrides})


# ---------------------------------------------------------------- facts


def test_facts_are_preformatted(aapl_sections) -> None:
    facts = ai.build_facts(AAPL, **aapl_sections)
    latest = facts["financials"]["quarters_newest_first"][0]
    assert latest["revenue"].startswith("$") and latest["revenue"].endswith("B")
    assert latest["gross_margin"].endswith("%")
    assert facts["snapshot"]["price"].startswith("$")


def test_facts_list_failed_sections(aapl_sections) -> None:
    sections = {**aapl_sections, "trends": SectionResult.failure("down", "test")}
    facts = ai.build_facts(AAPL, **sections)
    assert "price_trends" not in facts
    assert {"price_trends", "recent_news"} <= set(facts["unavailable_sections"])


def test_etf_facts_use_profile_not_financials(spy_info, spy_holdings_df) -> None:
    profile = parse_etf_profile(spy_info, parse_top_holdings(spy_holdings_df))
    facts = ai.build_facts(SPY, snapshot=ok(parse_snapshot(spy_info)), etf=ok(profile))
    assert "financials" not in facts and "financials" not in facts["unavailable_sections"]
    assert facts["etf_profile"]["expense_ratio"] == "0.095%"
    assert "wall_street" not in facts["unavailable_sections"]  # not expected for ETFs
    assert not {"market_cap", "sector", "industry"} & facts["snapshot"].keys()  # company-only fields


# ---------------------------------------------------------------- grounding check


def test_grounded_figures_pass(aapl_sections) -> None:
    facts = ai.build_facts(AAPL, **aapl_sections)
    q = facts["financials"]["quarters_newest_first"][0]
    analysis = sample_analysis(
        summary=f"Revenue was {q['revenue']} ({q['revenue_yoy']} YoY) with a {q['gross_margin']} gross margin.",
        bull_case=[f"Trailing P/E is {facts['financials']['trailing_pe']}."],
        bear_case=["The 50-day average matters in 2026."],  # bare integers are labels, not checked
    )
    assert ai.unverified_figures(analysis, facts) == []


def test_invented_figures_are_flagged(aapl_sections) -> None:
    facts = ai.build_facts(AAPL, **aapl_sections)
    analysis = sample_analysis(summary="Services revenue hit $27.3B, up 13.7%.")
    assert ai.unverified_figures(analysis, facts) == ["$27.3B", "13.7%"]


# ---------------------------------------------------------------- model call


class FakeBetaMessages:
    def __init__(self, parsed=None, stop_reason="end_turn", error=None) -> None:
        self.parsed, self.stop_reason, self.error, self.kwargs = parsed, stop_reason, error, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason)


def fake_llm(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeBetaMessages(**kwargs)))


def test_request_sends_schema_effort_and_fallback() -> None:
    llm = fake_llm(parsed=sample_analysis())
    ai.request_analysis({"x": 1}, llm, "claude-sonnet-5-5", 8000, "medium")
    sent = llm.beta.messages.kwargs
    assert sent["output_format"] is ai.Analysis
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["fallbacks"] == "default" and sent["betas"] == [ai.FALLBACK_BETA]
    assert sent["messages"][0]["content"].startswith("<facts>")


@pytest.mark.parametrize(
    "stop_reason, message", [("refusal", "declined"), ("max_tokens", "token limit"), ("end_turn", "no analysis")]
)
def test_request_handles_bad_stops(stop_reason: str, message: str) -> None:
    with pytest.raises(ai.DataUnavailableError, match=message):
        ai.request_analysis({}, fake_llm(parsed=None, stop_reason=stop_reason), "m", 100, "low")


def test_section_success_reports_unverified(aapl_sections) -> None:
    llm = fake_llm(parsed=sample_analysis(summary="Margins hit 99.9%."))
    result = ai.get_ai_analysis(AAPL, llm=llm, **aapl_sections)
    assert result.ok
    assert result.data.unverified_numbers == ["99.9%"]
    assert "not investment advice" in result.data.disclaimer


def test_section_api_error_is_friendly(aapl_sections) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=request), body=None)
    result = ai.get_ai_analysis(AAPL, llm=fake_llm(error=error), **aapl_sections)
    assert not result.ok and "rate-limiting" in result.error


def test_section_needs_core_data() -> None:
    result = ai.get_ai_analysis(AAPL, llm=fake_llm(parsed=sample_analysis()))
    assert not result.ok and "Not enough data" in result.error


def test_benchmark_is_not_compared_with_itself(spy_info) -> None:
    import pandas as pd
    from brief.trends import build_price_trends

    idx = pd.bdate_range("2025-01-01", periods=504, tz="America/New_York")
    hist = pd.DataFrame({"Close": [100 * 1.001**i for i in range(504)], "Volume": [1e6] * 504}, index=idx)
    trends = ok(build_price_trends(hist, hist, "SPY"))
    facts = ai.build_facts(SPY, snapshot=ok(parse_snapshot(spy_info)), trends=trends)
    assert "performance_vs_benchmark" not in facts["price_trends"]
    assert "excess" not in str(facts["price_trends"]["performance"])


def test_risk_facts_included(aapl_sections) -> None:
    import math

    import pandas as pd
    from brief.risk import build_risk_profile

    idx = pd.bdate_range("2024-10-01", periods=504, tz="America/New_York")
    a = pd.DataFrame({"Close": [100 * (1 + 0.02 * math.sin(i / 5)) * 1.0006**i for i in range(504)]}, index=idx)
    b = pd.DataFrame({"Close": [100 * (1 + 0.01 * math.sin(i / 5)) * 1.0003**i for i in range(504)]}, index=idx)
    risk = ok(build_risk_profile(a, b, "SPY", 0.0405, "AAPL"))
    facts = ai.build_facts(AAPL, risk=risk, **aapl_sections)
    rp = facts["risk_profile"]
    assert rp["risk_free_rate_13_week_tbill"] == "4.05%"
    assert {"beta_2y", "down_capture", "max_drawdown_2y"} <= rp["AAPL"].keys()
    assert "beta_2y" not in rp["SPY"] or rp["SPY"]["beta_2y"] == "1.00"
    assert "risk_profile" not in facts["unavailable_sections"]


def test_earnings_history_facts(aapl_sections) -> None:
    from brief.finnhub_client import parse_earnings_history
    from tests.conftest import load_json

    h = ok(parse_earnings_history(load_json("finnhub_earnings_history_aapl.json")))
    facts = ai.build_facts(AAPL, earnings_history=h, **aapl_sections)
    et = facts["earnings_track_record"]
    assert et["summary"].startswith("Beat estimates in 3 of the last 4")
    assert et["quarters_newest_first"][0]["result"] == "in line"
    assert "GAAP" in et["basis_note"]
