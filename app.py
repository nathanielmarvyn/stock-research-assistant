"""TickerBrief: Streamlit entry point.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from brief.config import get_settings
from brief.market_data import AssetType, TickerValidationError, first_number, normalize_symbol
from brief.models import DataUnavailableError
from brief.signals import build_signals
from brief.symbols import resolve_query
from brief.ui import graphics, loaders
from brief.ui import sections as ui

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="TickerBrief",
    page_icon=":material/candlestick_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUGGESTIONS = ("AAPL", "MSFT", "NVDA", "JPM", "SPY", "QQQ")
TOP_SUGGESTIONS = SUGGESTIONS[:4]  # fewer chips fit beside the search box
DISCLAIMER = (
    "For informational and educational purposes only. Not investment advice. Data may be delayed or "
    "incomplete; verify independently before making any investment decision."
)
SEARCH_PLACEHOLDER = "Search a company or ticker, e.g. Apple or AAPL"

# Layout and motion. Keyed containers get a ".st-key-<key>" class, which is what these target.
STYLES = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 3.75rem; }  /* clears Streamlit's header strip */
@keyframes tb-fade-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes tb-slide-in { from { opacity: 0; transform: translateX(-16px); } to { opacity: 1; transform: none; } }
.st-key-hero { animation: tb-fade-up .55s ease-out both; }
.st-key-topbar { animation: tb-slide-in .4s ease-out both; padding-bottom: .5rem;
  border-bottom: 1px solid rgba(128,128,128,.25); margin-bottom: .75rem; }
.st-key-brief_body { animation: tb-fade-up .5s .1s ease-out both; }
/* Selectors use ARIA roles, which are stabler than Streamlit's generated class names. */
.st-key-hero_search [role="group"] { min-height: 3.25rem; border-radius: 999px; padding-left: .9rem; }
.st-key-hero_search input[role="combobox"] { font-size: 1.05rem; }
.st-key-hero_picks { display: flex; justify-content: center; }  /* the chip group sizes to its content */
.tb-tagline { text-align: center; opacity: .8; font-size: 1.05rem; margin: .25rem 0 1.25rem; }
.tb-footnote { text-align: center; opacity: .65; font-size: .8rem; margin-top: 2.5rem; }
@media (prefers-reduced-motion: reduce) { .st-key-hero, .st-key-topbar, .st-key-brief_body { animation: none; } }
</style>
"""


# ---------------------------------------------------------------- navigation


def go_to(symbol: str | None) -> None:
    """Open a brief. The ticker lives in the URL (?ticker=AAPL), so briefs are shareable."""
    if symbol:
        st.query_params["ticker"] = symbol


def on_search(key: str) -> None:
    """Search box callback: resolve a picked label, ticker, or company name, then clear the box."""
    choice = st.session_state.get(key)
    if choice:
        go_to(resolve_query(choice, loaders.load_universe()))
    st.session_state[key] = None


def on_pick(key: str) -> None:
    """Suggestion chip callback."""
    go_to(st.session_state.get(key))
    st.session_state[key] = None


def go_home() -> None:
    """Back to the search page."""
    st.query_params.clear()


def search_box(key: str, placeholder: str = SEARCH_PLACEHOLDER) -> None:
    """Search-as-you-type over ~11,000 tickers and company names; free text is accepted too."""
    st.selectbox(
        "Search a company or ticker",
        [entry.label for entry in loaders.load_universe()],
        index=None,
        placeholder=placeholder,
        key=key,
        label_visibility="collapsed",
        accept_new_options=True,  # brand-new tickers still work; "apple" + Enter resolves by name
        filter_mode="contains",  # keeps list order, so well-known companies match first
        on_change=on_search,
        args=(key,),
    )


def suggestions(key: str, options: tuple[str, ...] = SUGGESTIONS) -> None:
    """Quick-pick chips."""
    st.pills("Popular", options, key=key, label_visibility="collapsed", on_change=on_pick, args=(key,))


# ---------------------------------------------------------------- views


def home() -> None:
    """Centered logo, wordmark, search box, and suggestions. No sidebar."""
    with st.container(key="hero"):
        st.html('<div style="height:14vh"></div>')
        _, middle, _ = st.columns([1, 2.4, 1])
        with middle:
            st.html(graphics.brand_html("large", ui.palette()))
            st.html(
                '<p class="tb-tagline">One-page research briefs for US stocks and ETFs: fundamentals, '
                "risk, insider activity, news, and a grounded AI analysis.</p>"
            )
            search_box("hero_search")
            suggestions("hero_picks")
            st.html(f'<p class="tb-footnote">{DISCLAIMER}</p>')


def top_bar() -> None:
    """Compact header for a brief: home, brand, search, and suggestions."""
    with st.container(key="topbar"):
        home_col, brand_col, search_col, picks_col = st.columns([1.2, 1.9, 3.3, 2.8], vertical_alignment="center")
        home_col.button("Home", icon=":material/home:", key="home", help="Back to search", on_click=go_home)
        with brand_col:
            st.html(graphics.brand_html("small", ui.palette()))
        with search_col:
            search_box("top_search", "Search company or ticker")
        with picks_col:
            suggestions("top_picks", TOP_SUGGESTIONS)


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


def market_pe() -> float | None:
    """The benchmark's trailing P/E, for the valuation signal."""
    try:
        return first_number(loaders.load_ticker(get_settings().benchmark_ticker)[1], "trailingPE")
    except (TickerValidationError, DataUnavailableError):
        return None


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

    # Reserve the "at a glance" slot near the top; it's filled once every section has loaded.
    st.divider()
    balance_slot = st.container()

    financials = wall_street = earnings_history = ownership = None
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

    with balance_slot:
        signals = build_signals(
            ticker,
            trends=trends,
            financials=financials,
            risk=risk,
            wall_street=wall_street,
            earnings_history=earnings_history,
            ownership=ownership,
            news=news,
            market_pe=market_pe(),
        )
        ui.render_signal_balance(signals)

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
    """App entry point: the search page, or a brief when ?ticker= is set."""
    st.html(STYLES)
    ticker = st.query_params.get("ticker", "").strip()
    # One shared slot for both views: the first element of the new view replaces the old
    # one immediately, instead of the previous page lingering, greyed out, while a brief loads.
    with st.empty().container():
        if ticker:
            top_bar()
            with st.container(key="brief_body"):
                brief(ticker)
        else:
            home()


main()
