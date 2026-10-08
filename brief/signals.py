"""Signal balance: a transparent, rule-based bull/bear reading of the brief.

Each factor maps one fact already shown on the page to a score from -1
(bearish) to +1 (bullish) with a written rule. The net reading is the
weighted average of the available scores. No model is involved, so the same
data always gives the same result, and the viewer can re-weight factors.

This summarizes the evidence in the brief. It is not a rating or a
recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from brief import formatting as fmt
from brief.financials import Financials
from brief.finnhub_client import EarningsHistory, WallStreetView
from brief.market_data import AssetType, TickerInfo
from brief.models import SectionResult
from brief.news import NewsBrief
from brief.ownership import OwnershipActivity
from brief.risk import RiskProfile
from brief.trends import PriceTrends

BULLISH_ABOVE = 0.33
BEARISH_BELOW = -0.33


@dataclass(frozen=True)
class Signal:
    """One factor's reading."""

    key: str
    name: str
    value: str  # the fact, as shown to the user
    score: float  # -1 bearish .. +1 bullish
    rule: str  # how the score was set


def clip(x: float, low: float = -1.0, high: float = 1.0) -> float:
    """Bound a score to [-1, 1]."""
    return max(low, min(high, x))


def net_score(signals: list[Signal], weights: dict[str, float]) -> float | None:
    """Weighted average of scores; factors missing from ``weights`` count once."""
    total = sum(weights.get(s.key, 1.0) for s in signals)
    if not signals or total <= 0:
        return None
    return sum(weights.get(s.key, 1.0) * s.score for s in signals) / total


def balance_label(net: float | None) -> str:
    """Plain-language zone for a net score."""
    if net is None:
        return "Not enough data"
    if net > BULLISH_ABOVE:
        return "Mostly bullish signals"
    if net < BEARISH_BELOW:
        return "Mostly bearish signals"
    return "Mixed signals"


# ---------------------------------------------------------------- factor rules


def trend_signal(t: PriceTrends) -> Signal | None:
    """Above both moving averages = +1, below both = -1, split = 0."""
    if t.above_sma50 is None or t.above_sma200 is None:
        return None
    score = 1.0 if t.above_sma50 and t.above_sma200 else -1.0 if not (t.above_sma50 or t.above_sma200) else 0.0
    where = {1.0: "above both", -1.0: "below both", 0.0: "between"}[score]
    return Signal("trend", "Price trend", f"Price {where} its 50- and 200-day averages", score,
                  "+1 above both averages, -1 below both, 0 in between.")


def relative_signal(t: PriceTrends, symbol: str) -> Signal | None:
    """1Y return vs. the benchmark: +15 points or more = +1, -15 or worse = -1."""
    if symbol == t.benchmark:
        return None
    one_year = next((p for p in t.performance if p.period == "1Y"), None)
    if one_year is None or one_year.excess_return is None:
        return None
    return Signal("relative", f"1Y return vs. {t.benchmark}",
                  f"{fmt.pct(one_year.excess_return, 1, signed=True)} vs. {t.benchmark}",
                  clip(one_year.excess_return / 0.15),
                  "Scaled so beating the market by 15 points or more scores +1; lagging by 15 or more, -1.")


def rsi_signal(t: PriceTrends) -> Signal | None:
    """Overbought (>70) leans bearish, oversold (<30) leans bullish; otherwise neutral."""
    if t.rsi14 is None:
        return None
    score = -0.5 if t.rsi14 > 70 else 0.5 if t.rsi14 < 30 else 0.0
    return Signal("rsi", "RSI (14-day)", f"{t.rsi14:.1f}", score,
                  "-0.5 above 70 (stretched), +0.5 below 30 (washed out), 0 otherwise.")


def revenue_signal(f: Financials) -> Signal | None:
    """Latest-quarter revenue growth: +10% YoY or more = +1, -10% or worse = -1."""
    yoy = f.quarters[0].revenue_yoy if f.quarters else None
    if yoy is None:
        return None
    return Signal("revenue", "Revenue growth", f"{fmt.pct(yoy, 1, signed=True)} YoY", clip(yoy / 0.10),
                  "Scaled so +10% year-over-year growth or more scores +1; a 10% decline, -1.")


def eps_signal(f: Financials) -> Signal | None:
    """Latest-quarter EPS growth: +15% YoY or more = +1, -15% or worse = -1."""
    yoy = f.quarters[0].eps_yoy if f.quarters else None
    if yoy is None:
        return None
    return Signal("eps", "EPS growth", f"{fmt.pct(yoy, 1, signed=True)} YoY", clip(yoy / 0.15),
                  "Scaled so +15% year-over-year EPS growth or more scores +1; a 15% decline, -1.")


def valuation_signal(f: Financials, market_pe: float | None) -> Signal | None:
    """Trailing P/E relative to the S&P 500's: 50% cheaper = +1, 50% richer = -1."""
    if f.trailing_pe is None or market_pe is None or f.trailing_pe <= 0 or market_pe <= 0:
        return None
    ratio = f.trailing_pe / market_pe
    return Signal("valuation", "Valuation", f"P/E {f.trailing_pe:.1f} vs. S&P 500 {market_pe:.1f}",
                  clip((1 - ratio) / 0.5),
                  "Compares trailing P/E with the S&P 500's: 50% cheaper or more scores +1; 50% richer or more, -1.")


def consensus_signal(w: WallStreetView) -> Signal | None:
    """Analyst score 1.5 (Strong Buy) = +1, 3 (Hold) = 0, 4.5 (Sell) = -1."""
    if w.consensus is None:
        return None
    c = w.consensus
    return Signal("consensus", "Analyst consensus", f"{c.label} ({c.score:.2f} of 5)", clip((3 - c.score) / 1.5),
                  "Maps the 1 (Strong Buy) to 5 (Strong Sell) score: 1.5 or better = +1, 3 = 0, 4.5 or worse = -1.")


def target_signal(w: WallStreetView) -> Signal | None:
    """Average price target upside: +15% = +1, -15% = -1."""
    if w.price_target is None or w.price_target.upside is None:
        return None
    up = w.price_target.upside
    return Signal("target", "Price target", f"{fmt.pct(up, 1, signed=True)} vs. price", clip(up / 0.15),
                  "Scaled so an average target 15% or more above the price scores +1; 15% below, -1.")


def earnings_signal(h: EarningsHistory) -> Signal | None:
    """(Beats - misses) / quarters."""
    n = len(h.quarters)
    if not n:
        return None
    beats, misses = h.count("beat"), h.count("miss")
    return Signal("earnings", "Earnings track record", f"{beats} beats, {misses} misses in {n} quarters",
                  (beats - misses) / n, "Beats minus misses, divided by the number of quarters.")


def sharpe_signal(r: RiskProfile) -> Signal | None:
    """1Y Sharpe minus the benchmark's: +1.0 or more = +1."""
    t, b = r.ticker, r.benchmark
    if b is None or t.sharpe_1y is None or b.sharpe_1y is None:
        return None
    return Signal("sharpe", "Risk-adjusted return", f"Sharpe {t.sharpe_1y:.2f} vs. {b.sharpe_1y:.2f}",
                  clip(t.sharpe_1y - b.sharpe_1y),
                  "Difference between its 1Y Sharpe ratio and the market's, capped at ±1.")


def downside_signal(r: RiskProfile) -> Signal | None:
    """Down capture 50% = +1, 100% = 0, 150% = -1."""
    dc = r.ticker.down_capture
    if r.benchmark is None or dc is None:
        return None
    return Signal("downside", "Downside capture", f"{fmt.pct(dc, 0)} of market declines", clip((1 - dc) / 0.5),
                  "Losing half as much as the market in down months scores +1; matching it, 0; 1.5x, -1.")


def insider_signal(o: OwnershipActivity) -> Signal | None:
    """Buying flags are strong; large sales are a weak negative, since sales are often routine."""
    a = o.insiders
    if a is None:
        return None
    kinds = {f.kind for f in a.flags}
    if "cluster_buying" in kinds:
        score, value = 1.0, "Clustered insider buying"
    elif "executive_purchase" in kinds:
        score, value = 0.5, "Executive open-market purchase"
    elif "large_sale" in kinds:
        score, value = -0.25, "Large insider sales"
    elif a.trades and a.net_value < 0:
        score, value = -0.1, "Net insider selling"
    elif a.trades:
        score, value = 0.25, "Net insider buying"
    else:
        score, value = 0.0, "No open-market insider trades"
    return Signal("insiders", "Insider activity", value, score,
                  "+1 clustered buying, +0.5 executive purchase, +0.25 net buying, -0.1 net selling, "
                  "-0.25 large sales (sales are weak signals: often pre-scheduled).")


def news_signal(n: NewsBrief) -> Signal | None:
    """The overall news sentiment score, already on -1..+1."""
    if n.overall_score is None:
        return None
    return Signal("news", "News sentiment", f"{n.overall_label} ({n.overall_score:+.2f})", n.overall_score,
                  "Average of headline labels: positive +1, neutral 0, negative -1.")


# ---------------------------------------------------------------- assembly


def _data(result: SectionResult | None):
    return result.data if result is not None and result.ok else None


def build_signals(
    ticker: TickerInfo,
    trends: SectionResult[PriceTrends] | None = None,
    financials: SectionResult[Financials] | None = None,
    risk: SectionResult[RiskProfile] | None = None,
    wall_street: SectionResult[WallStreetView] | None = None,
    earnings_history: SectionResult[EarningsHistory] | None = None,
    ownership: SectionResult[OwnershipActivity] | None = None,
    news: SectionResult[NewsBrief] | None = None,
    market_pe: float | None = None,
) -> list[Signal]:
    """Every factor that has data, in display order. Missing sections are simply skipped."""
    candidates: list[Signal | None] = []
    if t := _data(trends):
        candidates += [trend_signal(t), relative_signal(t, ticker.symbol), rsi_signal(t)]
    if ticker.asset_type is AssetType.STOCK:
        if f := _data(financials):
            candidates += [revenue_signal(f), eps_signal(f), valuation_signal(f, market_pe)]
        if h := _data(earnings_history):
            candidates.append(earnings_signal(h))
        if w := _data(wall_street):
            candidates += [consensus_signal(w), target_signal(w)]
        if o := _data(ownership):
            candidates.append(insider_signal(o))
    if r := _data(risk):
        candidates += [sharpe_signal(r), downside_signal(r)]
    if n := _data(news):
        candidates.append(news_signal(n))
    return [s for s in candidates if s is not None]
