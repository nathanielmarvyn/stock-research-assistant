"""Stock Research Assistant: Streamlit entry point.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from brief.config import get_settings
from brief.market_data import AssetType, TickerValidationError, normalize_symbol
from brief.models import DataUnavailableError
from brief.ui import loaders
from brief.ui import sections as ui

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Stock Research Assistant", page_icon=":material/query_stats:", layout="wide")

EXAMPLES = ("AAPL", "MSFT", "JPM", "SPY", "QQQ")
DISCLAIMER = (
    "For informational and educational purposes only. Not investment advice. Data may be delayed or "
    "incomplete; verify independently before making any investment decision."
)


def sidebar() -> None:
    """Ticker form and examples; the chosen ticker lives in the URL (?ticker=AAPL) so briefs are shareable."""
    with st.sidebar:
        st.header("Research a ticker", anchor=False)
        with st.form("ticker_form", border=False):
            raw = st.text_input("Ticker symbol", value=st.query_params.get("ticker", ""), placeholder="e.g. AAPL")
            if st.form_submit_button("Generate brief", type="primary", width="stretch") and raw.strip():
                st.query_params["ticker"] = raw.strip().upper()
                st.rerun()
        example = st.pills("Examples", EXAMPLES, key="example")
        if example and example != st.session_state.get("last_example"):
            st.session_state["last_example"] = example
            st.query_params["ticker"] = example
            st.rerun()
        st.divider()
        st.caption(
            "US-listed stocks and ETFs. Data from Yahoo Finance and Finnhub, cached for "
            f"{get_settings().cache_ttl_seconds // 60} minutes. Summaries and analysis by Claude."
        )
        st.caption(DISCLAIMER)


def landing() -> None:
    """Shown before a ticker is chosen."""
    st.title("Stock Research Assistant", anchor=False)
    st.markdown(
        "Enter a US stock or ETF ticker to get a one-page research brief: a snapshot, the last four "
        "quarters of financials, price trends against the S&P 500, the Wall Street view, recent news "
        "with sentiment, and an AI-written bull/bear analysis grounded in that data."
    )
    st.info("Pick an example in the sidebar or type a ticker to begin.", icon=":material/arrow_back:")


def ai_allowance(symbol: str) -> str | None:
    """Return a message if this session has used up its AI analyses, else record the ticker and return None."""
    used: set[str] = st.session_state.setdefault("ai_tickers", set())
    limit = get_settings().max_ai_analyses_per_session
    if symbol in used or len(used) < limit:
        used.add(symbol)
        return None
    return (
        f"This demo runs AI analysis for up to {limit} different tickers per session to control API costs. "
        "Every other section above is still live."
    )


def brief(raw: str) -> None:
    """Render the full brief for one ticker. Each section fails independently."""
    try:
        symbol = normalize_symbol(raw)
        with st.spinner(f"Looking up {symbol}…"):
            ticker, _ = loaders.load_ticker(symbol)
    except (TickerValidationError, DataUnavailableError) as exc:
        st.error(str(exc), icon=":material/error:")
        st.caption("Check the symbol and try again. Mutual funds, crypto, and non-US listings aren't supported.")
        return

    is_etf = ticker.asset_type is AssetType.ETF
    snapshot = loaders.load_snapshot(symbol)
    ui.render_header(ticker, snapshot)

    with st.spinner("Loading market data…"):
        etf = loaders.load_etf_profile(symbol) if is_etf else None
        earnings = None if is_etf else loaders.load_earnings(symbol)
    ui.render_snapshot(snapshot, earnings, etf)

    st.divider()
    if is_etf:
        ui.render_etf_profile(etf)
    else:
        with st.spinner("Loading financials…"):
            financials = loaders.load_financials(symbol)
        ui.render_financials(financials)

    st.divider()
    with st.spinner("Loading price history…"):
        trends = loaders.load_trends(symbol)
    ui.render_trends(trends, symbol)

    st.divider()
    with st.spinner("Measuring risk…"):
        risk = loaders.load_risk(symbol)
    ui.render_risk(risk, symbol)

    if not is_etf:
        st.divider()
        with st.spinner("Loading analyst ratings…"):
            wall_street = loaders.load_wall_street(symbol)
        ui.render_wall_street(wall_street, snapshot.data.price if snapshot.ok else None)

        st.divider()
        with st.spinner("Loading earnings history…"):
            earnings_history = loaders.load_earnings_history(symbol)
        ui.render_earnings_history(earnings_history)

        st.divider()
        with st.spinner("Loading ownership and insider activity…"):
            ownership = loaders.load_ownership(symbol)
        ui.render_ownership(ownership)

    st.divider()
    with st.spinner("Loading news and scoring sentiment…"):
        news = loaders.load_news(symbol)
    ui.render_news(news)

    st.divider()
    limit_message = ai_allowance(symbol)
    analysis = None
    if limit_message is None:
        with st.spinner("Writing AI analysis…"):
            analysis = loaders.load_analysis(symbol)
    ui.render_analysis(analysis, limit_message)

    st.divider()
    st.caption(f"Brief generated {ui.generated_at()}. {DISCLAIMER}")


def main() -> None:
    """App entry point."""
    sidebar()
    ticker = st.query_params.get("ticker", "").strip()
    if ticker:
        brief(ticker)
    else:
        landing()


main()
