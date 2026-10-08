"""Print the brief sections built so far for a ticker (developer preview, no UI).

Usage:  python scripts/preview.py AAPL
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brief.financials import get_financials  # noqa: E402
from brief.market_data import (  # noqa: E402
    AssetType,
    TickerValidationError,
    get_etf_profile,
    get_snapshot,
    validate_ticker,
)
from brief.models import DataUnavailableError, SectionResult  # noqa: E402
from brief.trends import get_price_trends  # noqa: E402


def fmt(value: float | None, spec: str) -> str:
    """Format a number, or a dash when it's missing."""
    return "—" if value is None else format(value, spec)


def big(value: float | None) -> str:
    """Format a large dollar amount as $1.23T / $4.5B / $6.7M."""
    if value is None:
        return "—"
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"${value / size:,.2f}{suffix}"
    return f"${value:,.0f}"


def header(title: str, result: SectionResult) -> bool:
    """Print a section header; return False (and print the error) if it failed."""
    print(f"\n== {title}  (as of {result.as_of:%Y-%m-%d %H:%M} UTC, {result.source})")
    if not result.ok:
        print(f"   ! {result.error}")
    return result.ok


def main(raw: str) -> None:
    """Validate the ticker and print each available section."""
    try:
        ticker, info = validate_ticker(raw)
    except (TickerValidationError, DataUnavailableError) as exc:
        print(f"Invalid ticker: {exc}")
        return
    print(f"{ticker.symbol} — {ticker.name} ({ticker.asset_type.value.upper()}, {ticker.exchange})")

    snap = get_snapshot(info)
    if header("Snapshot", snap):
        s = snap.data
        print(f"   Price {fmt(s.price, ',.2f')}  {fmt(s.day_change, '+.2f')} ({fmt(s.day_change_pct, '+.2%')})")
        print(f"   52-week range {fmt(s.week52_low, ',.2f')} – {fmt(s.week52_high, ',.2f')}")
        print(f"   Market cap {big(s.market_cap)}  Dividend yield {fmt(s.dividend_yield, '.2%')}")
        if s.sector:
            print(f"   {s.sector} / {s.industry}")

    trends = get_price_trends(ticker.symbol)
    if header("Price trends", trends):
        t = trends.data
        print(f"   {'Period':<7}{ticker.symbol:>9}{t.benchmark:>9}{'Excess':>9}")
        for p in t.performance:
            print(f"   {p.period:<7}{fmt(p.ticker_return, '+.1%'):>9}{fmt(p.benchmark_return, '+.1%'):>9}"
                  f"{fmt(p.excess_return, '+.1%'):>9}")
        print(f"   50-day SMA {fmt(t.sma50, ',.2f')} ({'above' if t.above_sma50 else 'below'})  "
              f"200-day SMA {fmt(t.sma200, ',.2f')} ({'above' if t.above_sma200 else 'below'})")
        print(f"   RSI(14) {fmt(t.rsi14, '.1f')}  Volume {fmt(t.latest_volume, ',.0f')} vs 30-day avg "
              f"{fmt(t.avg_volume_30d, ',.0f')} ({fmt(t.relative_volume, '.2f')}x)")

    if ticker.asset_type is AssetType.ETF:
        etf = get_etf_profile(ticker.symbol, info)
        if header("ETF profile", etf):
            e = etf.data
            print(f"   Expense ratio {fmt(e.expense_ratio, '.3%')}  AUM {big(e.aum)}  {e.category or ''}")
            for h in e.top_holdings[:10]:
                print(f"   {h.symbol:<6} {h.name:<30} {h.weight:6.2%}")
        return

    fin = get_financials(ticker.symbol, info)
    if header("Financials", fin):
        f = fin.data
        print(f"   P/E {fmt(f.trailing_pe, '.1f')}  Fwd P/E {fmt(f.forward_pe, '.1f')}  D/E {fmt(f.debt_to_equity, '.2f')}")
        print(f"   {'Quarter':<11}{'Revenue':>10}{'YoY':>8}{'Net inc':>10}{'EPS':>7}{'YoY':>8}"
              f"{'Gross':>7}{'Oper':>7}{'Net':>7}{'FCF':>10}")
        for q in f.quarters:
            print(f"   {q.period_end:%Y-%m-%d} {big(q.revenue):>10}{fmt(q.revenue_yoy, '+.1%'):>8}"
                  f"{big(q.net_income):>10}{fmt(q.eps, '.2f'):>7}{fmt(q.eps_yoy, '+.1%'):>8}"
                  f"{fmt(q.gross_margin, '.1%'):>7}{fmt(q.operating_margin, '.1%'):>7}"
                  f"{fmt(q.net_margin, '.1%'):>7}{big(q.free_cash_flow):>10}")
        for note in f.notes:
            print(f"   Note: {note}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
