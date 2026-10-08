"""Streamlit render functions, one per brief section.

Each function takes a SectionResult, prints a header with its "data as of"
timestamp and source, and either renders the data or a warning card. None of
them raise, so one bad section can't stop the page.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from brief import formatting as fmt
from brief.ai_analysis import AnalysisResult
from brief.financials import Financials
from brief.finnhub_client import EPS_BASIS_NOTE, EarningsEvent, EarningsHistory, WallStreetView
from brief.market_data import EtfProfile, Snapshot, TickerInfo
from brief.models import SectionResult
from brief import ownership as own
from brief.news import NewsBrief
from brief.ownership import InsiderActivity, OwnershipActivity, OwnershipBreakdown
from brief.risk import Drawdown, RiskProfile, risk_summary
from brief.signals import Signal, balance_label, net_score
from brief.trends import PriceTrends
from brief.ui import charts

EASTERN = ZoneInfo("America/New_York")
SENTIMENT_BADGES = {  # icon + label, never color alone
    "positive": (":material/trending_up:", "Positive", "green"),
    "neutral": (":material/trending_flat:", "Neutral", "gray"),
    "negative": (":material/trending_down:", "Negative", "red"),
}


# ---------------------------------------------------------------- shared pieces


def md(text: str) -> str:
    """Escape '$' so Streamlit markdown doesn't render '$1 ... $2' as a LaTeX formula."""
    return text.replace("$", r"\$")


def bullets(points: list[str]) -> str:
    """Markdown bullet list with '$' escaped."""
    return md("\n".join(f"- {p}" for p in points))


def palette() -> charts.Palette:
    """Chart colors for the viewer's current light/dark theme."""
    try:
        return charts.DARK if st.context.theme.type == "dark" else charts.LIGHT
    except Exception:  # outside a running app (tests, scripts)
        return charts.LIGHT


def as_of_text(result: SectionResult, date_only: bool = False) -> str:
    """'Oct 7, 2026, 4:00 PM ET', or just the date for period-end dates.

    Date-only values (e.g. a quarter end) are not timezone-converted: midnight
    UTC on Jun 30 would otherwise display as Jun 29 in New York.
    """
    if date_only:
        d = result.as_of
        return f"{d:%b} {d.day}, {d.year}"
    local = result.as_of.astimezone(EASTERN)
    return f"{local:%b} {local.day}, {local.year}, {local:%I:%M %p}".replace(" 0", " ") + " ET"


def section(title: str, result: SectionResult, as_of_label: str | None = None) -> bool:
    """Section heading plus provenance caption; shows a warning and returns False on failure."""
    st.subheader(title, anchor=False)
    if not result.ok:
        st.warning(f"{title} unavailable: {result.error}", icon=":material/info:")
        return False
    st.caption(f"Data as of {as_of_label or as_of_text(result)} · Source: {result.source}")
    return True


def range_position(low: float | None, high: float | None, price: float | None) -> float | None:
    """Where the price sits in its 52-week range, 0 (low) to 1 (high)."""
    if None in (low, high, price) or high == low:
        return None
    return min(max((price - low) / (high - low), 0.0), 1.0)


# ---------------------------------------------------------------- signal balance

MIN_SIGNALS = 3


def _reset_weights(keys: list[str]) -> None:
    for key in keys:
        st.session_state[f"weight_{key}"] = 1.0
        st.session_state.pop(f"slider_{key}", None)  # sliders re-read the stored weight


def _store_weight(key: str) -> None:
    """Copy a slider's value into a plain session key that outlives the slider widget."""
    st.session_state[f"weight_{key}"] = st.session_state[f"slider_{key}"]


@st.fragment
def render_signal_balance(signals: list[Signal]) -> None:
    """Rule-based bull/bear balance with adjustable weights. Reruns on its own when a slider moves."""
    st.subheader("Signal balance", anchor=False)
    st.caption(
        "A rule-based summary of the evidence in this brief, not a rating or recommendation. "
        "Each factor scores from −1 (bearish) to +1 (bullish); hover a bar to see its rule."
    )
    if len(signals) < MIN_SIGNALS:
        st.info("Not enough data loaded to summarize the signals.", icon=":material/info:")
        return

    for s in signals:
        st.session_state.setdefault(f"weight_{s.key}", 1.0)
    weights = {s.key: float(st.session_state[f"weight_{s.key}"]) for s in signals}
    net = net_score(signals, weights)
    if net is None:
        st.info("All factors are weighted at zero. Raise at least one weight.", icon=":material/info:")
    else:
        bulls = sum(1 for s in signals if s.score > 0 and weights[s.key] > 0)
        bears = sum(1 for s in signals if s.score < 0 and weights[s.key] > 0)
        neutral = sum(1 for s in signals if weights[s.key] > 0) - bulls - bears
        st.markdown(
            f"**{balance_label(net)}** · net {net:+.2f} · {bulls} bullish · {bears} bearish · {neutral} neutral"
        )
        st.plotly_chart(charts.balance_meter(net, palette()), width="stretch", config={"displayModeBar": False})
    st.plotly_chart(charts.signal_bars(signals, weights, palette()), width="stretch", config={"displayModeBar": False})

    # A keyed toggle (not an expander): its state survives the fragment rerun each slider
    # move triggers, so the panel stays open while the viewer adjusts weights.
    if st.toggle("Adjust weights", key="show_signal_weights"):
        with st.container(border=True):
            st.caption("Set how much each factor counts. 0 leaves it out; 3x triples its influence.")
            cols = st.columns(3)
            for i, s in enumerate(signals):
                # Weights live in "weight_*" keys, not the slider's own key: Streamlit drops widget
                # state when the panel is closed, which would otherwise reset the weights.
                cols[i % 3].slider(
                    s.name, 0.0, 3.0, value=weights[s.key], step=0.5, key=f"slider_{s.key}", format="%.1fx",
                    help=s.rule, on_change=_store_weight, args=(s.key,),
                )
            st.button("Reset to equal weights", on_click=_reset_weights, args=([s.key for s in signals],),
                      icon=":material/restart_alt:")


# ---------------------------------------------------------------- 1. snapshot


def render_header(ticker: TickerInfo, snapshot: SectionResult[Snapshot]) -> None:
    """Company name (to confirm the ticker), identifiers, and headline price."""
    st.title(ticker.name, anchor=False)
    details = [ticker.symbol, "ETF" if ticker.asset_type.value == "etf" else "Stock", ticker.exchange or ""]
    if snapshot.ok and snapshot.data.sector:
        details += [snapshot.data.sector, snapshot.data.industry or ""]
    st.caption(" · ".join(d for d in details if d))


def render_snapshot(
    snapshot: SectionResult[Snapshot],
    earnings: SectionResult[EarningsEvent] | None,
    etf: SectionResult[EtfProfile] | None,
) -> None:
    """Section 1: price, size, range, and income metrics."""
    if not section("Snapshot", snapshot):
        return
    s = snapshot.data
    # Two rows of two (not one row of four) so values aren't truncated on narrower screens.
    top, bottom = st.columns(2), st.columns(2)
    cols = [top[0], top[1], bottom[0], bottom[1]]
    cols[0].metric(
        "Price",
        fmt.money(s.price),
        f"{fmt.money(s.day_change)} ({fmt.pct(s.day_change_pct, 2, signed=True)})" if s.day_change is not None else None,
        border=True,
    )
    if etf is not None:
        e = etf.data if etf.ok else None
        cols[1].metric("Assets (AUM)", fmt.money(e.aum if e else None), border=True)
        cols[2].metric("Expense ratio", fmt.pct(e.expense_ratio if e else None, 3), border=True)
    else:
        cols[1].metric("Market cap", fmt.money(s.market_cap), border=True)
        if earnings is not None and earnings.ok:
            ev = earnings.data
            cols[2].metric(
                "Next earnings",
                f"{ev.date:%b} {ev.date.day}",
                f"{ev.date.year} · {ev.timing or 'time TBA'}",
                delta_color="off",
                delta_arrow="off",
                help=f"EPS estimate {fmt.num(ev.eps_estimate)}, revenue estimate "
                f"{fmt.money(ev.revenue_estimate)}. Source: Finnhub.",
                border=True,
            )
        else:
            cols[2].metric("Next earnings", "Not announced", border=True)
    cols[3].metric("Dividend yield", fmt.pct(s.dividend_yield, 2), border=True)

    position = range_position(s.week52_low, s.week52_high, s.price)
    with st.container(border=True):
        st.markdown(md(f"**52-week range** · {fmt.money(s.week52_low)} – {fmt.money(s.week52_high)}"))
        if position is not None:
            st.progress(position, text=f"Price is at {position:.0%} of its 52-week range")


# ---------------------------------------------------------------- ETF profile


def render_etf_profile(etf: SectionResult[EtfProfile]) -> None:
    """ETF alternative to company financials: category and top holdings."""
    if not section("Fund profile", etf):
        return
    e = etf.data
    st.markdown(md(
        f"**Category:** {e.category or fmt.MISSING} · **Fund family:** {e.fund_family or fmt.MISSING} · "
        f"**Expense ratio:** {fmt.pct(e.expense_ratio, 3)} · **AUM:** {fmt.money(e.aum)}"
    ))
    if not e.top_holdings:
        st.info("Holdings data isn't available for this fund.")
        return
    top = e.top_holdings[:10]
    st.markdown(f"**Top {len(top)} holdings** ({sum(h.weight for h in top):.1%} of assets)")
    st.plotly_chart(charts.holdings_chart(top, palette()), width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------- 2. financials


def financials_table(f: Financials) -> pd.DataFrame:
    """Statement-style table: metrics as rows, quarters (newest first) as columns."""
    rows = {
        "Revenue": [fmt.money(q.revenue) for q in f.quarters],
        "Revenue YoY": [fmt.pct(q.revenue_yoy, signed=True) for q in f.quarters],
        "Net income": [fmt.money(q.net_income) for q in f.quarters],
        "Net income YoY": [fmt.pct(q.net_income_yoy, signed=True) for q in f.quarters],
        "EPS (diluted)": [fmt.num(q.eps) for q in f.quarters],
        "EPS YoY": [fmt.pct(q.eps_yoy, signed=True) for q in f.quarters],
        "Gross margin": [fmt.pct(q.gross_margin) for q in f.quarters],
        "Operating margin": [fmt.pct(q.operating_margin) for q in f.quarters],
        "Net margin": [fmt.pct(q.net_margin) for q in f.quarters],
        "Free cash flow": [fmt.money(q.free_cash_flow) for q in f.quarters],
        "Free cash flow YoY": [fmt.pct(q.free_cash_flow_yoy, signed=True) for q in f.quarters],
    }
    columns = [f"Qtr ended {q.period_end:%b} {q.period_end.day}, {q.period_end.year}" for q in f.quarters]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def render_financials(financials: SectionResult[Financials]) -> None:
    """Section 2: valuation ratios and the last four quarters."""
    if not section("Financials", financials, f"latest quarter ended {as_of_text(financials, date_only=True)}"):
        return
    f = financials.data
    cols = st.columns(3)
    cols[0].metric("P/E (trailing)", fmt.num(f.trailing_pe, 1), border=True,
                   help="Price ÷ last 12 months' EPS. Blank when earnings are negative.")
    cols[1].metric("P/E (forward)", fmt.num(f.forward_pe, 1), border=True,
                   help="Price ÷ analysts' estimated EPS for the next 12 months.")
    cols[2].metric("Debt / equity", fmt.num(f.debt_to_equity, 2, "x"), border=True,
                   help="Total debt ÷ shareholders' equity, latest balance sheet. Blank if equity is negative.")
    st.dataframe(financials_table(f), width="stretch")
    st.caption("YoY compares each quarter with the same quarter a year earlier; — means that quarter isn't available.")
    for note in f.notes:
        st.info(note, icon=":material/info:")


# ---------------------------------------------------------------- 3. price trends


def rsi_zone(rsi: float | None) -> str:
    """Conventional RSI reading."""
    if rsi is None:
        return ""
    return "Overbought (>70)" if rsi > 70 else "Oversold (<30)" if rsi < 30 else "Neutral (30–70)"


def render_trends(trends: SectionResult[PriceTrends], symbol: str) -> None:
    """Section 3: chart, performance vs. benchmark, moving averages, RSI, volume."""
    if not section("Price trends", trends):
        return
    t = trends.data
    st.plotly_chart(charts.price_chart(t.chart, symbol, palette()), width="stretch", config={"displayModeBar": False})

    is_benchmark = symbol == t.benchmark
    perf = pd.DataFrame(
        {
            "Period": [p.period for p in t.performance],
            symbol: [fmt.pct(p.ticker_return, signed=True) for p in t.performance],
            **(
                {}
                if is_benchmark
                else {
                    f"{t.benchmark} (S&P 500)": [fmt.pct(p.benchmark_return, signed=True) for p in t.performance],
                    "Difference": [fmt.pct(p.excess_return, signed=True) for p in t.performance],
                }
            ),
        }
    )
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Total return** (includes dividends)" + ("" if is_benchmark else f" vs. {t.benchmark}"))
        st.dataframe(perf, hide_index=True, width="stretch")
    with right:
        def position(above: bool | None) -> str:
            return "" if above is None else "price above" if above else "price below"

        c1, c2 = st.columns(2)
        c1.metric("50-day avg", fmt.money(t.sma50), position(t.above_sma50), delta_color="off", delta_arrow="off")
        c2.metric("200-day avg", fmt.money(t.sma200), position(t.above_sma200), delta_color="off", delta_arrow="off")
        c3, c4 = st.columns(2)
        c3.metric("RSI (14-day)", fmt.num(t.rsi14, 1), rsi_zone(t.rsi14), delta_color="off", delta_arrow="off",
                  help="Relative Strength Index (Wilder). Above 70 is conventionally 'overbought', below 30 'oversold'.")
        c4.metric(
            "Volume vs 30-day avg" + (" (prior session)" if t.volume_from_prior_session else ""),
            fmt.num(t.relative_volume, 2, "x"),
            f"{fmt.volume(t.latest_volume)} vs {fmt.volume(t.avg_volume_30d)}",
            delta_color="off",
            delta_arrow="off",
            help="Today's session is still trading, so this compares the last full session's volume."
            if t.volume_from_prior_session
            else "Latest session's volume vs. the average of the 30 sessions before it.",
        )


# ---------------------------------------------------------------- 3b. risk profile


def short_date(ts: pd.Timestamp | None) -> str:
    """'Apr 8, 2025' from a timestamp."""
    return "—" if ts is None else f"{ts:%b} {ts.day}, {ts.year}"


def drawdown_text(d: Drawdown | None) -> str:
    """'-33.4%' depth for the table cell."""
    return fmt.MISSING if d is None else fmt.pct(d.depth, 1)


def drawdown_detail(d: Drawdown | None) -> str:
    """When the worst fall happened and whether it has recovered."""
    if d is None or d.depth == 0:
        return "No decline from a high in this window."
    recovery = f"recovered by {short_date(d.recovery_date)}" if d.recovery_date is not None else "not yet recovered"
    return f"Worst fall from a high: {short_date(d.peak_date)} to {short_date(d.trough_date)}, {recovery}."


def risk_table(p: RiskProfile, symbol: str) -> pd.DataFrame:
    """Metric rows with the ticker, the benchmark (if different), and a plain-English meaning."""
    t, b, bench = p.ticker, p.benchmark, p.benchmark_symbol
    rf = f" ({fmt.pct(p.risk_free_rate, 2)} T-bill)" if p.risk_free_rate is not None else ""
    beta_meaning = (
        f"Tends to move about {t.beta_2y:.2f}% when the market moves 1%." if t.beta_2y is not None
        else "Sensitivity to market moves (1.00 = moves with the market)."
    )
    rows: list[tuple[str, str, str, str]] = [
        ("Volatility (1Y, annualized)", fmt.pct(t.volatility_1y), fmt.pct(b.volatility_1y) if b else "",
         "Typical size of price swings over a year."),
        ("Total return (1Y)", fmt.pct(t.return_1y, signed=True), fmt.pct(b.return_1y, signed=True) if b else "",
         "Price change plus dividends."),
        ("Sharpe ratio (1Y)", fmt.num(t.sharpe_1y), fmt.num(b.sharpe_1y) if b else "",
         f"Return above the risk-free rate{rf}, per unit of volatility. Higher is better."),
        ("Max drawdown (1Y)", drawdown_text(t.drawdown_1y), drawdown_text(b.drawdown_1y) if b else "",
         drawdown_detail(t.drawdown_1y)),
        ("Max drawdown (2Y)", drawdown_text(t.drawdown_2y), drawdown_text(b.drawdown_2y) if b else "",
         drawdown_detail(t.drawdown_2y)),
    ]
    if b is not None:
        rows += [
            ("Beta (2Y)", fmt.num(t.beta_2y), "1.00", beta_meaning),
            ("Correlation (2Y)", fmt.num(t.correlation_2y), "1.00",
             "1.00 = moves in lockstep with the market; lower adds more diversification."),
            ("Up capture", fmt.pct(t.up_capture, 0), "100%",
             f"Share of the market's gains captured in its up months ({p.capture_months} months)."),
            ("Down capture", fmt.pct(t.down_capture, 0), "100%",
             "Share of the market's losses taken in its down months. Lower is better."),
        ]
    columns = ["Metric", symbol] + ([bench] if b is not None else []) + ["What it means"]
    return pd.DataFrame(
        [(m, v, bv, why) if b is not None else (m, v, why) for m, v, bv, why in rows], columns=columns
    )


def render_risk(risk: SectionResult[RiskProfile], symbol: str) -> None:
    """Section 3b: how much risk the price history shows, next to the market's."""
    if not section("Risk profile", risk):
        return
    p = risk.data
    if summary := risk_summary(p, symbol):
        st.markdown(md(summary))
    # st.table (not st.dataframe) so the explanations wrap instead of being cut off.
    st.table(risk_table(p, symbol).set_index("Metric"))
    st.markdown("**Drawdown from previous high** (2Y)")
    st.plotly_chart(charts.drawdown_chart(p.underwater, palette()), width="stretch", config={"displayModeBar": False})
    st.caption(
        "Past volatility and drawdowns describe history, not future risk. Beta and correlation use 2 years "
        "of daily returns; capture ratios use complete months."
    )


# ---------------------------------------------------------------- 4. Wall Street


def render_wall_street(view: SectionResult[WallStreetView], price: float | None) -> None:
    """Section 4: analyst consensus and average price target."""
    if not section("Wall Street view", view):
        return
    w = view.data
    left, right = st.columns(2)
    with left:
        if c := w.consensus:
            change = None if c.prior_score is None else c.score - c.prior_score
            st.metric(
                f"Analyst consensus · {c.period:%b %Y}",
                c.label,
                None if change is None or abs(change) < 0.005 else
                f"{'more bullish' if change < 0 else 'more bearish'} than last month ({c.prior_score:.2f} → {c.score:.2f})",
                delta_color="off",
                delta_arrow="off",
                help="Weighted average of ratings on a 1 (Strong Buy) to 5 (Strong Sell) scale. Source: Finnhub.",
                border=True,
            )
            st.plotly_chart(charts.ratings_chart(c, palette()), width="stretch", config={"displayModeBar": False})
            st.caption(f"{c.total} analysts · score {c.score:.2f} on a 1 (Strong Buy) to 5 (Strong Sell) scale")
        else:
            st.info("Analyst ratings aren't available.")
    with right:
        if p := w.price_target:
            st.metric(
                "Average 12-month price target",
                fmt.money(p.mean),
                f"{fmt.pct(p.upside, signed=True)} vs. current price" if p.upside is not None else None,
                border=True,
                help="Mean of analysts' published targets. Source: Yahoo Finance.",
            )
            st.caption(md(f"Range {fmt.money(p.low)} – {fmt.money(p.high)} · {p.analyst_count or '—'} analysts"))
        else:
            st.info("No price target coverage.")


# ---------------------------------------------------------------- 4b. earnings track record

OUTCOME_LABELS = {  # icon + word, never color alone
    "beat": "▲ Beat",
    "miss": "▼ Miss",
    "in line": "● In line",
}


def earnings_table(h: EarningsHistory) -> pd.DataFrame:
    """Quarter rows, newest first: estimate, actual, surprise, outcome."""
    return pd.DataFrame(
        {
            "Quarter ended": [f"{q.period_end:%b} {q.period_end.day}, {q.period_end.year}" for q in h.quarters],
            "Fiscal qtr": [f"Q{q.fiscal_quarter} {q.fiscal_year}" if q.fiscal_quarter else "—" for q in h.quarters],
            "EPS estimate": [fmt.money(q.estimate) for q in h.quarters],
            "EPS actual": [fmt.money(q.actual) for q in h.quarters],
            "Surprise": [fmt.pct(q.surprise_pct, 1, signed=True) for q in h.quarters],
            "Result": [OUTCOME_LABELS.get(q.outcome or "", "—") for q in h.quarters],
        }
    )


def render_earnings_history(history: SectionResult[EarningsHistory]) -> None:
    """Section 4b: how reported EPS compared with expectations."""
    if not section("Earnings track record", history, f"latest quarter ended {as_of_text(history, date_only=True)}"):
        return
    h = history.data
    st.markdown(h.summary())
    # Stacked (not side by side) so all six columns stay readable on narrow screens.
    st.table(earnings_table(h).set_index("Quarter ended"))
    st.plotly_chart(charts.earnings_chart(h, palette()), width="stretch", config={"displayModeBar": False})
    st.caption(f"Beat or miss means a surprise of at least ±1%; smaller differences count as in line. {EPS_BASIS_NOTE}")


# ---------------------------------------------------------------- 4c. ownership & insiders

TRADE_LABELS = {"buy": "▲ Buy", "sell": "▼ Sell"}  # icon + word, never color alone
MAX_INSIDER_ROWS = 12


def insider_table(a: InsiderActivity) -> pd.DataFrame:
    """Open-market trades, newest first, one row per person per day."""
    return pd.DataFrame(
        {
            "Date": [f"{t.trade_date:%b} {t.trade_date.day}, {t.trade_date.year}" for t in a.trades],
            "Insider": [t.name for t in a.trades],
            "Role": [t.role or "—" for t in a.trades],
            "Trade": [TRADE_LABELS[t.side] for t in a.trades],
            "Shares": [f"{t.shares:,}" for t in a.trades],
            "Value": [fmt.money_compact(t.value) for t in a.trades],
            "% of holdings": [fmt.pct(t.stake_change, 1) for t in a.trades],
        }
    )


def holders_table(o: OwnershipBreakdown) -> pd.DataFrame:
    """Top institutional holders with their latest quarter-over-quarter change."""
    return pd.DataFrame(
        {
            "Holder": [h.name for h in o.top_holders],
            "% of shares": [fmt.pct(h.pct_held, 2) for h in o.top_holders],
            "Shares": [fmt.volume(h.shares) for h in o.top_holders],
            "Value": [fmt.money(h.value, 1) for h in o.top_holders],
            "Change vs. prior quarter": [fmt.pct(h.pct_change, 1, signed=True) for h in o.top_holders],
        }
    )


def routine_text(counts: dict[str, int]) -> str:
    """'Not counted as trades: 18 option exercises, 13 grants/awards, 7 tax withholding, 3 gifts.'"""
    if not counts:
        return ""
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return "Not counted as buying or selling: " + ", ".join(f"{n} {label}" for label, n in items) + "."


def render_ownership(result: SectionResult[OwnershipActivity]) -> None:
    """Section 4c: who owns the stock and what insiders have done with it."""
    if not section("Ownership & insider activity", result):
        return
    a, o = result.data.insiders, result.data.ownership

    cols = st.columns(3)
    cols[0].metric("Held by insiders", fmt.pct(o.insider_pct if o else None, 2), border=True)
    cols[1].metric(
        "Held by institutions",
        fmt.pct(o.institution_pct if o else None, 1),
        f"{o.institution_count:,} institutions" if o and o.institution_count else None,
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    if a is not None:
        net = a.net_value
        cols[2].metric(
            "Net insider trades (6M)",
            fmt.money_compact(net) if a.trades else "None",
            ("net selling" if net < 0 else "net buying") if a.trades else "no open-market trades",
            delta_color="off",
            delta_arrow="off",
            help="Open-market purchases minus sales (SEC codes P and S). Grants, option exercises, "
            "tax withholding, and gifts are excluded because they aren't investment decisions.",
            border=True,
        )

    for note in result.data.notes:
        st.info(note, icon=":material/info:")

    if a is not None:
        st.markdown(f"**Insider trades** · {md(a.summary())}")
        for flag in a.flags:
            icon = ":material/trending_up:" if flag.tone == "positive" else ":material/warning:"
            (st.success if flag.tone == "positive" else st.warning)(f"**{flag.title}.** {md(flag.detail)}", icon=icon)
        if a.trades:
            shown = insider_table(a).head(MAX_INSIDER_ROWS)
            st.table(shown.set_index("Date"))
            if len(a.trades) > MAX_INSIDER_ROWS:
                st.caption(f"Showing the {MAX_INSIDER_ROWS} most recent of {len(a.trades)} trades; totals include all.")
        if routine := routine_text(a.routine_counts):
            st.caption(routine)
        st.caption(md(
            f"Flags: {own.CLUSTER_MIN_BUYERS}+ insiders buying within {own.CLUSTER_WINDOW_DAYS} days; any open-market "
            f"buy by a CEO, CFO, COO, president, or chair; sales of ${own.LARGE_SALE_VALUE / 1e6:.0f}M+ in a day, or "
            f"{own.LARGE_SALE_STAKE:.0%}+ of a person's holdings (when at least ${own.LARGE_SALE_STAKE_MIN_VALUE / 1e6:.0f}M)."
        ))

    if o is not None and o.top_holders:
        reported = {h.date_reported for h in o.top_holders if h.date_reported}
        as_of = f" as of {max(reported):%b} {max(reported).day}, {max(reported).year}" if reported else ""
        st.markdown(f"**Top institutional holders**{as_of}")
        st.table(holders_table(o).set_index("Holder"))
        st.caption(
            "From 13F filings, which institutions submit up to 45 days after each quarter ends, so "
            "holdings can be several months old."
        )


# ---------------------------------------------------------------- 5. news


def render_news(news: SectionResult[NewsBrief]) -> None:
    """Section 5: headlines with summaries and sentiment."""
    if not section("Recent news", news):
        return
    n = news.data
    if n.overall_label:
        st.metric(
            "Overall news sentiment",
            n.overall_label,
            f"{n.overall_score:+.2f} on a −1 to +1 scale",
            delta_color="off",
            delta_arrow="off",
            help="Average of per-headline labels (positive = +1, neutral = 0, negative = −1).",
            border=True,
        )
    for note in (n.sentiment_note, n.coverage_note):
        if note:
            st.info(note, icon=":material/info:")
    for h in n.headlines:
        with st.container(border=True):
            st.markdown(f"**[{md(h.headline)}]({h.url})**")
            st.caption(f"{h.source} · {h.published.astimezone(EASTERN):%b} {h.published.astimezone(EASTERN).day}")
            if h.summary:
                st.markdown(md(h.summary))
            if h.sentiment:
                icon, label, color = SENTIMENT_BADGES[h.sentiment]
                st.badge(label, icon=icon, color=color)


# ---------------------------------------------------------------- 6. AI analysis


def render_analysis(analysis: SectionResult[AnalysisResult] | None, limit_message: str | None = None) -> None:
    """Section 6: model-written analysis, grounding warnings, and disclaimer."""
    if analysis is None:
        st.subheader("AI analysis", anchor=False)
        st.info(limit_message or "AI analysis not run.", icon=":material/info:")
        return
    if not section("AI analysis", analysis):
        return
    a, result = analysis.data.analysis, analysis.data
    st.markdown(md(a.summary))
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**:material/trending_up: Bull case**")
        st.markdown(bullets(a.bull_case))
    with right, st.container(border=True):
        st.markdown("**:material/trending_down: Bear case**")
        st.markdown(bullets(a.bear_case))
    left, right = st.columns(2)
    with left:
        st.markdown("**Key risks**")
        st.markdown(bullets(a.key_risks))
    with right:
        st.markdown("**What to watch**")
        st.markdown(bullets(a.what_to_watch))
    if result.unverified_numbers:
        st.warning(
            md("These figures in the analysis couldn't be matched to the data above. Treat them with caution: "
               + ", ".join(result.unverified_numbers)),
            icon=":material/warning:",
        )
    st.caption(f"Generated by {result.model} from the data in this brief only.")
    st.info(result.disclaimer, icon=":material/gavel:")


def generated_at() -> str:
    """Footer timestamp."""
    now = datetime.now(EASTERN)
    return f"{now:%b} {now.day}, {now.year}, {now:%I:%M %p}".replace(" 0", " ") + " ET"
