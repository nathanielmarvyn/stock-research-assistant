"""Tests for brief.financials calculations and statement parsing."""

import pandas as pd
import pytest

from brief.models import DataUnavailableError

from brief.financials import (
    QUARTERS_SHOWN,
    debt_to_equity,
    parse_financials,
    safe_ratio,
    yoy_change,
)


# ---------------------------------------------------------------- calculations


def test_yoy_change_basic() -> None:
    assert yoy_change(110, 100) == pytest.approx(0.10)


def test_yoy_change_from_loss_reads_as_improvement() -> None:
    assert yoy_change(-5, -10) == pytest.approx(0.5)


@pytest.mark.parametrize("cur, prior", [(None, 1), (1, None), (1, 0)])
def test_yoy_change_undefined(cur, prior) -> None:
    assert yoy_change(cur, prior) is None


def test_safe_ratio() -> None:
    assert safe_ratio(25, 100) == 0.25
    assert safe_ratio(25, 0) is None
    assert safe_ratio(None, 100) is None


def test_debt_to_equity_negative_equity_is_undefined() -> None:
    assert debt_to_equity(50, 100) == 0.5
    assert debt_to_equity(50, -10) is None


# ---------------------------------------------------------------- parsing


def test_aapl_quarters(aapl_statements, aapl_info) -> None:
    fin = parse_financials(*aapl_statements, aapl_info)
    assert len(fin.quarters) == QUARTERS_SHOWN
    periods = [q.period_end for q in fin.quarters]
    assert periods == sorted(periods, reverse=True)  # newest first

    latest = fin.quarters[0]
    income = aapl_statements[0]
    rev = income.iloc[:, 0]["Total Revenue"]
    assert latest.revenue == rev
    assert latest.gross_margin == pytest.approx(income.iloc[:, 0]["Gross Profit"] / rev)
    assert latest.net_margin == pytest.approx(income.iloc[:, 0]["Net Income"] / rev)
    assert latest.eps == income.iloc[:, 0]["Diluted EPS"]


def test_aapl_yoy_only_where_prior_year_exists(aapl_statements, aapl_info) -> None:
    fin = parse_financials(*aapl_statements, aapl_info)
    income = aapl_statements[0]
    # Fixture has 5 quarters: only the newest has a same-quarter-last-year match.
    expected = income.iloc[:, 0]["Total Revenue"] / income.iloc[:, 4]["Total Revenue"] - 1
    assert fin.quarters[0].revenue_yoy == pytest.approx(expected)
    assert all(q.revenue_yoy is None for q in fin.quarters[1:])


def test_aapl_valuation_and_leverage(aapl_statements, aapl_info) -> None:
    fin = parse_financials(*aapl_statements, aapl_info)
    balance = aapl_statements[2]
    expected_de = balance.iloc[:, 0]["Total Debt"] / balance.iloc[:, 0]["Stockholders Equity"]
    assert fin.debt_to_equity == pytest.approx(expected_de)
    assert fin.trailing_pe == aapl_info["trailingPE"]
    assert fin.quarters[0].free_cash_flow is not None


def test_missing_rows_become_none() -> None:
    """A bank-like statement with no gross profit and no cash flow still parses."""
    income = pd.DataFrame(
        {pd.Timestamp("2026-06-30"): [100.0, 20.0]},
        index=["Total Revenue", "Net Income"],
    )
    fin = parse_financials(income, None, None, {"debtToEquity": 150.0})
    q = fin.quarters[0]
    assert q.gross_margin is None and q.free_cash_flow is None
    assert q.net_margin == pytest.approx(0.2)
    assert fin.debt_to_equity == pytest.approx(1.5)  # info fallback, percent -> ratio


def test_fcf_falls_back_to_ocf_plus_capex() -> None:
    period = pd.Timestamp("2026-06-30")
    income = pd.DataFrame({period: [100.0]}, index=["Total Revenue"])
    cashflow = pd.DataFrame(
        {period: [30.0, -5.0]}, index=["Operating Cash Flow", "Capital Expenditure"]
    )
    fin = parse_financials(income, cashflow, None, {})
    assert fin.quarters[0].free_cash_flow == 25.0


def test_empty_income_statement_raises() -> None:
    with pytest.raises(DataUnavailableError):
        parse_financials(pd.DataFrame(), None, None, {})


@pytest.mark.parametrize(
    "industry, expected",
    [
        ("Banks - Diversified", False),
        ("Insurance - Life", False),
        ("Credit Services", True),  # Visa/Mastercard: FCF is meaningful
        ("Consumer Electronics", True),
        (None, True),
    ],
)
def test_fcf_is_meaningful(industry, expected) -> None:
    from brief.financials import fcf_is_meaningful

    assert fcf_is_meaningful(industry) is expected


def test_bank_fcf_hidden_with_note(aapl_statements) -> None:
    from brief.financials import FCF_NOTE

    fin = parse_financials(*aapl_statements, {"industry": "Banks - Diversified"})
    assert all(q.free_cash_flow is None for q in fin.quarters)
    assert fin.notes == (FCF_NOTE,)
