"""Company financials: last four quarters, margins, YoY changes, and valuation ratios.

Same pattern as ``market_data``: ``fetch_statements`` does the network I/O and
``parse_financials`` is a pure function tested offline against saved statements.
All ratios are fractions (0.25 == 25%).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from brief.market_data import SOURCE, DataSourceError, first_number
from brief.models import DataUnavailableError, SectionResult, safe_section

QUARTERS_SHOWN = 4
# Quarter-end dates drift a few days between years; match "one year ago" loosely.
_YOY_TOLERANCE_DAYS = 20

# yfinance row labels vary by company type; first match wins.
_REVENUE = ("Total Revenue", "Operating Revenue")
_GROSS_PROFIT = ("Gross Profit",)
_OPERATING_INCOME = ("Operating Income", "Total Operating Income As Reported")
_NET_INCOME = ("Net Income", "Net Income Common Stockholders")
_EPS = ("Diluted EPS", "Basic EPS")
_FCF = ("Free Cash Flow",)
_OCF = ("Operating Cash Flow",)
_CAPEX = ("Capital Expenditure",)
_DEBT = ("Total Debt",)
_EQUITY = ("Stockholders Equity", "Common Stock Equity")

# For banks and insurers, operating cash flow mixes in deposits, loans, and
# premiums, so "free cash flow" swings wildly and says little about the business.
_FCF_NOT_MEANINGFUL_INDUSTRIES = ("Banks", "Insurance")
FCF_NOTE = (
    "Free cash flow is omitted: for banks and insurers, operating cash flow "
    "includes deposit, loan, and premium flows, so FCF isn't a meaningful measure."
)


@dataclass(frozen=True)
class Quarter:
    """One fiscal quarter's results. YoY fields compare to the same quarter a year earlier."""

    period_end: datetime
    revenue: float | None
    net_income: float | None
    eps: float | None
    free_cash_flow: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    revenue_yoy: float | None
    net_income_yoy: float | None
    eps_yoy: float | None
    free_cash_flow_yoy: float | None


@dataclass(frozen=True)
class Financials:
    """Financials section: recent quarters (newest first) plus valuation ratios."""

    quarters: list[Quarter]
    trailing_pe: float | None
    forward_pe: float | None
    debt_to_equity: float | None  # ratio, e.g. 0.78 means $0.78 debt per $1 equity
    notes: tuple[str, ...] = ()  # caveats to show alongside the numbers


# ---------------------------------------------------------------- calculations


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """numerator / denominator, or None if either is missing or the denominator is 0."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def yoy_change(current: float | None, prior: float | None) -> float | None:
    """Fractional change vs. a prior value.

    Divides by ``abs(prior)`` so a move from a loss to a smaller loss reads as
    an improvement (-10 -> -5 is +50%, not -50%).
    """
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior)


def debt_to_equity(total_debt: float | None, equity: float | None) -> float | None:
    """Debt / equity. None when equity is zero or negative, where the ratio is meaningless."""
    if total_debt is None or equity is None or equity <= 0:
        return None
    return total_debt / equity


# ---------------------------------------------------------------- statement access


def fcf_is_meaningful(industry: str | None) -> bool:
    """False for banks and insurers, where free cash flow is distorted by balance-sheet flows."""
    return not (industry and industry.startswith(_FCF_NOT_MEANINGFUL_INDUSTRIES))


def _value(df: pd.DataFrame | None, labels: tuple[str, ...], col: Any) -> float | None:
    """Look up the first available row label for one statement column."""
    if df is None or df.empty or col not in df.columns:
        return None
    for label in labels:
        if label in df.index:
            value = df.at[label, col]
            if pd.notna(value):
                return float(value)
    return None


def _matching_column(df: pd.DataFrame | None, period_end: pd.Timestamp) -> Any | None:
    """Find the column in ``df`` for the same quarter end (within a few days)."""
    if df is None or df.empty:
        return None
    for col in df.columns:
        if abs((pd.Timestamp(col) - period_end).days) <= 5:
            return col
    return None


def _year_ago_column(columns: list[pd.Timestamp], period_end: pd.Timestamp) -> pd.Timestamp | None:
    """Return the column ~1 year before ``period_end``, if present."""
    target = period_end - pd.DateOffset(years=1)
    for col in columns:
        if abs((col - target).days) <= _YOY_TOLERANCE_DAYS:
            return col
    return None


def _free_cash_flow(cashflow: pd.DataFrame | None, period_end: pd.Timestamp) -> float | None:
    """FCF for a quarter: reported value, else operating cash flow + capex (capex is negative)."""
    col = _matching_column(cashflow, period_end)
    if col is None:
        return None
    fcf = _value(cashflow, _FCF, col)
    if fcf is not None:
        return fcf
    ocf, capex = _value(cashflow, _OCF, col), _value(cashflow, _CAPEX, col)
    return ocf + capex if ocf is not None and capex is not None else None


def _sorted_periods(income: pd.DataFrame) -> list[pd.Timestamp]:
    """Quarter-end columns, newest first, skipping quarters with no revenue reported."""
    periods = sorted((pd.Timestamp(c) for c in income.columns), reverse=True)
    return [p for p in periods if _value(income, _REVENUE, _matching_column(income, p)) is not None]


# ---------------------------------------------------------------- parsing


def parse_financials(
    income: pd.DataFrame,
    cashflow: pd.DataFrame | None,
    balance: pd.DataFrame | None,
    info: dict[str, Any],
) -> Financials:
    """Build the Financials section from quarterly statements and the info dict."""
    if income is None or income.empty:
        raise DataUnavailableError("No quarterly financial statements are available.")

    periods = _sorted_periods(income)
    show_fcf = fcf_is_meaningful(info.get("industry"))

    def figures(period: pd.Timestamp) -> dict[str, float | None]:
        col = _matching_column(income, period)
        return {
            "revenue": _value(income, _REVENUE, col),
            "gross_profit": _value(income, _GROSS_PROFIT, col),
            "operating_income": _value(income, _OPERATING_INCOME, col),
            "net_income": _value(income, _NET_INCOME, col),
            "eps": _value(income, _EPS, col),
            "fcf": _free_cash_flow(cashflow, period) if show_fcf else None,
        }

    quarters: list[Quarter] = []
    for period in periods[:QUARTERS_SHOWN]:
        cur = figures(period)
        prior_period = _year_ago_column(periods, period)
        prev = figures(prior_period) if prior_period is not None else {}
        quarters.append(
            Quarter(
                period_end=period.to_pydatetime().replace(tzinfo=timezone.utc),
                revenue=cur["revenue"],
                net_income=cur["net_income"],
                eps=cur["eps"],
                free_cash_flow=cur["fcf"],
                gross_margin=safe_ratio(cur["gross_profit"], cur["revenue"]),
                operating_margin=safe_ratio(cur["operating_income"], cur["revenue"]),
                net_margin=safe_ratio(cur["net_income"], cur["revenue"]),
                revenue_yoy=yoy_change(cur["revenue"], prev.get("revenue")),
                net_income_yoy=yoy_change(cur["net_income"], prev.get("net_income")),
                eps_yoy=yoy_change(cur["eps"], prev.get("eps")),
                free_cash_flow_yoy=yoy_change(cur["fcf"], prev.get("fcf")),
            )
        )

    # Prefer D/E from the latest balance sheet; fall back to yfinance's
    # info field, which is reported in percent units (78.4 == 0.784x).
    de = None
    if periods and balance is not None:
        col = _matching_column(balance, periods[0])
        de = debt_to_equity(_value(balance, _DEBT, col), _value(balance, _EQUITY, col))
    if de is None:
        info_de = first_number(info, "debtToEquity")
        de = info_de / 100 if info_de is not None else None

    return Financials(
        quarters=quarters,
        trailing_pe=first_number(info, "trailingPE"),
        forward_pe=first_number(info, "forwardPE"),
        debt_to_equity=de,
        notes=() if show_fcf else (FCF_NOTE,),
    )


# ---------------------------------------------------------------- fetching


def fetch_statements(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch quarterly income statement, cash flow, and balance sheet."""
    try:
        ticker = yf.Ticker(symbol)
        return (
            ticker.quarterly_income_stmt,
            ticker.quarterly_cashflow,
            ticker.quarterly_balance_sheet,
        )
    except YFRateLimitError as exc:
        raise DataSourceError(
            "Yahoo Finance is rate-limiting requests. Try again in a minute."
        ) from exc


@safe_section(SOURCE)
def get_financials(
    symbol: str,
    info: dict[str, Any],
    statements: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
) -> SectionResult[Financials]:
    """Financials section, stamped with the latest reported quarter end.

    ``statements`` can be passed in (e.g. from a cache); otherwise they are fetched.
    """
    income, cashflow, balance = statements or fetch_statements(symbol)
    financials = parse_financials(income, cashflow, balance, info)
    as_of = financials.quarters[0].period_end if financials.quarters else datetime.now(timezone.utc)
    return SectionResult.success(financials, SOURCE, as_of=as_of)
