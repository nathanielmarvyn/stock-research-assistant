"""Tests for the rule-based signal balance."""

from datetime import date

import pandas as pd
import pytest

from brief import signals as sig
from brief.financials import Financials, Quarter
from brief.finnhub_client import AnalystConsensus, EarningsHistory, EarningsResult, PriceTarget, WallStreetView
from brief.market_data import AssetType, TickerInfo
from brief.models import SectionResult
from brief.news import NewsBrief
from brief.ownership import Flag, InsiderActivity, InsiderTrade, OwnershipActivity
from brief.risk import RiskMetrics, RiskProfile
from brief.trends import PeriodPerformance, PriceTrends

AAPL = TickerInfo("AAPL", "Apple Inc.", AssetType.STOCK, "NMS", "USD")
SPY = TickerInfo("SPY", "SPDR S&P 500 ETF Trust", AssetType.ETF, "PCX", "USD")


def ok(data) -> SectionResult:
    return SectionResult.success(data, "test")


def trends(close=100.0, sma50=90.0, sma200=80.0, rsi=55.0, excess=0.15, benchmark="SPY") -> PriceTrends:
    perf = [PeriodPerformance("1Y", 0.10 + excess, 0.10)]
    return PriceTrends(close, perf, benchmark, sma50, sma200, rsi, 1e6, 1e6, 1.0, pd.DataFrame())


def quarter(rev_yoy=0.164, eps_yoy=0.287) -> Quarter:
    return Quarter(date(2026, 6, 30), 1e9, 1e8, 2.0, 1e8, 0.5, 0.3, 0.2, rev_yoy, 0.1, eps_yoy, None)


def metrics(sharpe, down=None) -> RiskMetrics:
    return RiskMetrics(0.2, 0.1, sharpe, None, None, 1.0, 0.6, 1.0, down)


# ---------------------------------------------------------------- individual rules


@pytest.mark.parametrize(
    "sma50, sma200, score", [(90.0, 80.0, 1.0), (110.0, 120.0, -1.0), (110.0, 80.0, 0.0)]
)
def test_trend_rule(sma50, sma200, score) -> None:
    assert sig.trend_signal(trends(sma50=sma50, sma200=sma200)).score == score


@pytest.mark.parametrize("excess, score", [(0.15, 1.0), (0.30, 1.0), (0.075, 0.5), (-0.15, -1.0)])
def test_relative_rule_scales_and_clips(excess, score) -> None:
    assert sig.relative_signal(trends(excess=excess), "AAPL").score == pytest.approx(score)


def test_relative_rule_skipped_for_benchmark_itself() -> None:
    assert sig.relative_signal(trends(), "SPY") is None


@pytest.mark.parametrize("rsi, score", [(75, -0.5), (25, 0.5), (55, 0.0)])
def test_rsi_rule(rsi, score) -> None:
    assert sig.rsi_signal(trends(rsi=rsi)).score == score


def test_growth_rules() -> None:
    f = Financials([quarter(rev_yoy=0.05, eps_yoy=-0.30)], 38.9, 35.4, 0.78)
    assert sig.revenue_signal(f).score == pytest.approx(0.5)
    assert sig.eps_signal(f).score == -1.0


@pytest.mark.parametrize("pe, market, score", [(38.9, 24.7, -1.0), (24.7, 24.7, 0.0), (18.5, 24.7, pytest.approx(0.502, abs=1e-3))])
def test_valuation_rule(pe, market, score) -> None:
    f = Financials([quarter()], pe, None, None)
    assert sig.valuation_signal(f, market).score == score


def test_valuation_skipped_for_negative_earnings() -> None:
    assert sig.valuation_signal(Financials([quarter()], -5.0, None, None), 24.7) is None


def test_wall_street_rules() -> None:
    c = AnalystConsensus(date(2026, 9, 1), 12, 22, 15, 3, 1, 2.23, "Buy", 2.13)
    w = WallStreetView(c, PriceTarget(328.0, 405.0, 215.0, 39, -0.031))
    assert sig.consensus_signal(w).score == pytest.approx((3 - 2.23) / 1.5)
    assert sig.target_signal(w).score == pytest.approx(-0.031 / 0.15)


def test_earnings_rule() -> None:
    qs = [EarningsResult(date(2026, m, 28), 1, 2026, 1.0, 1.0, s) for m, s in ((3, 0.05), (6, 0.04), (9, -0.05), (12, 0.0))]
    assert sig.earnings_signal(EarningsHistory(qs)).score == pytest.approx((2 - 1) / 4)


def test_risk_rules() -> None:
    r = RiskProfile(metrics(1.16, down=0.55), metrics(0.97), "SPY", 0.04, 23, pd.DataFrame())
    assert sig.sharpe_signal(r).score == pytest.approx(0.19)
    assert sig.downside_signal(r).score == pytest.approx(0.9)


def _activity(flags=(), trades=()) -> OwnershipActivity:
    return OwnershipActivity(InsiderActivity(list(trades), {}, list(flags), date(2026, 4, 8), date(2026, 10, 8)), None)


def test_insider_rule_priorities() -> None:
    cluster = Flag("cluster_buying", "positive", "t", "d")
    sale = Flag("large_sale", "caution", "t", "d")
    sell = InsiderTrade("A", "Officer", "sell", 100, 1e6, date(2026, 9, 1), 0.1)
    assert sig.insider_signal(_activity([cluster, sale])).score == 1.0  # buying outranks sales
    assert sig.insider_signal(_activity([sale], [sell])).score == -0.25
    assert sig.insider_signal(_activity([], [sell])).score == -0.1
    assert sig.insider_signal(_activity()).score == 0.0


# ---------------------------------------------------------------- assembly and net


def test_net_score_weighting() -> None:
    s = [sig.Signal("a", "A", "", 1.0, ""), sig.Signal("b", "B", "", -1.0, "")]
    assert sig.net_score(s, {}) == 0.0
    assert sig.net_score(s, {"a": 3.0}) == pytest.approx(0.5)  # (3 - 1) / 4
    assert sig.net_score(s, {"a": 0.0}) == -1.0
    assert sig.net_score(s, {"a": 0.0, "b": 0.0}) is None


@pytest.mark.parametrize("net, label", [(0.5, "Mostly bullish signals"), (-0.5, "Mostly bearish signals"), (0.1, "Mixed signals"), (None, "Not enough data")])
def test_balance_label(net, label) -> None:
    assert sig.balance_label(net) == label


def test_build_signals_stock_and_skips_failures() -> None:
    news = NewsBrief([], 0.1, "Neutral")
    out = sig.build_signals(
        AAPL,
        trends=ok(trends()),
        financials=ok(Financials([quarter()], 38.9, 35.4, 0.78)),
        risk=SectionResult.failure("down", "test"),
        news=ok(news),
        market_pe=24.7,
    )
    keys = [s.key for s in out]
    assert keys == ["trend", "relative", "rsi", "revenue", "eps", "valuation", "news"]


def test_build_signals_etf_ignores_company_factors() -> None:
    out = sig.build_signals(SPY, trends=ok(trends()), financials=ok(Financials([quarter()], 20.0, None, None)))
    assert [s.key for s in out] == ["trend", "rsi"]  # no relative (SPY vs itself), no fundamentals
