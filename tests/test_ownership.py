"""Tests for insider classification, flags, role matching, and ownership parsing."""

from datetime import date

import pandas as pd
import pytest

from brief import ownership as own
from tests.conftest import load_json

TODAY = date(2026, 10, 8)


def form4(name, code, change, price, when, held_after, derivative=False) -> dict:
    return {
        "name": name, "transactionCode": code, "change": change, "transactionPrice": price,
        "transactionDate": when, "share": held_after, "isDerivative": derivative,
    }


ROLES = {
    own._name_tokens("SMITH JANE A"): "Chief Executive Officer",
    own._name_tokens("DOE JOHN"): "Director",
    own._name_tokens("ROE RICK"): "Officer",
}


# ---------------------------------------------------------------- roles


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Smith Jane A", "Chief Executive Officer"),  # case
        ("Jane Smith", "Chief Executive Officer"),  # order + missing middle initial
        ("Doe John", "Director"),
        ("Someone Else", None),
    ],
)
def test_find_role(name, expected) -> None:
    assert own.find_role(name, ROLES) == expected


def test_role_lookup_from_yfinance_tables() -> None:
    tx = pd.DataFrame(load_json("yf_insider_transactions_aapl.json"))
    roster = pd.DataFrame(load_json("yf_insider_roster_aapl.json"))
    roles = own.build_role_lookup(tx, roster)
    assert own.find_role("Ternus John", roles) == "Chief Executive Officer"
    assert own.find_role("O'Brien Deirdre", roles) == "Officer"


# ---------------------------------------------------------------- classification


def test_only_open_market_trades_count() -> None:
    rows = [
        form4("Smith Jane A", "S", -10_000, 100.0, "2026-09-01", 90_000),
        form4("Smith Jane A", "A", 5_000, 0.0, "2026-09-02", 95_000),
        form4("Smith Jane A", "M", 2_000, 50.0, "2026-09-03", 97_000),
        form4("Smith Jane A", "F", -800, 100.0, "2026-09-03", 96_200),
        form4("Doe John", "P", 1_000, 100.0, "2026-09-04", 11_000, derivative=True),  # derivative: not a share buy
    ]
    a = own.parse_insider_activity(rows, ROLES, TODAY)
    assert [(t.name, t.side) for t in a.trades] == [("Smith, Jane A", "sell")]
    assert a.routine_counts == {
        "grants/awards": 1, "option exercises": 1, "tax withholding": 1, "derivative trades": 1,
    }


def test_same_day_slices_are_combined() -> None:
    # Three slices on one day; 'share' is holdings after each slice.
    rows = [
        form4("Smith Jane A", "S", -5_000, 100.0, "2026-10-02", 95_000),
        form4("Smith Jane A", "S", -10_000, 101.0, "2026-10-02", 85_000),
        form4("Smith Jane A", "S", -5_000, 102.0, "2026-10-02", 80_000),
    ]
    t = own.parse_insider_activity(rows, ROLES, TODAY).trades[0]
    assert t.shares == 20_000
    assert t.value == pytest.approx(5_000 * 100 + 10_000 * 101 + 5_000 * 102)
    assert t.stake_change == pytest.approx(20_000 / 100_000)  # held 100,000 before the first slice


def test_window_excludes_old_trades() -> None:
    rows = [form4("Doe John", "S", -1, 10.0, "2026-01-01", 10)]
    assert own.parse_insider_activity(rows, ROLES, TODAY).trades == []


def test_net_flow_and_summary() -> None:
    rows = [
        form4("Smith Jane A", "S", -10_000, 100.0, "2026-09-01", 90_000),
        form4("Doe John", "P", 2_000, 100.0, "2026-09-05", 12_000),
    ]
    a = own.parse_insider_activity(rows, ROLES, TODAY)
    assert a.value_sold == 1_000_000 and a.value_bought == 200_000
    assert a.net_value == -800_000
    assert a.summary() == "1 insider bought $0.2M; 1 insider sold $1.0M in the past 6 months."


# ---------------------------------------------------------------- flags


def test_cluster_buying_flag() -> None:
    rows = [
        form4("Smith Jane A", "P", 1_000, 50.0, "2026-08-01", 51_000),
        form4("Doe John", "P", 500, 50.0, "2026-08-10", 5_500),
        form4("Roe Rick", "P", 300, 50.0, "2026-08-25", 3_300),  # 24 days after the first buy
    ]
    kinds = [f.kind for f in own.parse_insider_activity(rows, ROLES, TODAY).flags]
    assert kinds.count("cluster_buying") == 1
    assert "executive_purchase" in kinds  # the CEO's buy


def test_no_cluster_when_buys_spread_out() -> None:
    rows = [
        form4("Doe John", "P", 500, 50.0, "2026-05-01", 5_500),
        form4("Roe Rick", "P", 300, 50.0, "2026-06-15", 3_300),
        form4("Other Person", "P", 300, 50.0, "2026-08-01", 3_300),
    ]
    assert not own.parse_insider_activity(rows, ROLES, TODAY).flags


@pytest.mark.parametrize(
    "change, price, held_after, flagged",
    [
        (-60_000, 100.0, 1_000_000, True),  # $6M: over the value threshold
        (-30_000, 40.0, 100_000, True),  # $1.2M and 23% of holdings
        (-30_000, 10.0, 100_000, False),  # 23% of holdings, but only $0.3M
        (-10_000, 100.0, 1_000_000, False),  # $1M and 1%: neither
    ],
)
def test_large_sale_flag(change, price, held_after, flagged) -> None:
    rows = [form4("Smith Jane A", "S", change, price, "2026-09-01", held_after)]
    kinds = [f.kind for f in own.parse_insider_activity(rows, ROLES, TODAY).flags]
    assert ("large_sale" in kinds) is flagged


def test_large_sale_applies_to_any_role() -> None:
    rows = [form4("Roe Rick", "S", -100_000, 100.0, "2026-09-01", 10_000)]  # $10M by a plain "Officer"
    assert [f.kind for f in own.parse_insider_activity(rows, ROLES, TODAY).flags] == ["large_sale"]


def test_one_large_sale_flag_per_person() -> None:
    rows = [
        form4("Doe John", "S", -60_000, 100.0, "2026-08-01", 900_000),
        form4("Doe John", "S", -70_000, 100.0, "2026-09-01", 830_000),
    ]
    flags = own.parse_insider_activity(rows, ROLES, TODAY).flags
    assert len(flags) == 1
    assert "130,000 shares ($13.0M" in flags[0].detail and "across 2 days" in flags[0].detail


def test_captured_aapl_insiders() -> None:
    tx = pd.DataFrame(load_json("yf_insider_transactions_aapl.json"))
    roster = pd.DataFrame(load_json("yf_insider_roster_aapl.json"))
    a = own.parse_insider_activity(load_json("finnhub_insiders_aapl.json"), own.build_role_lookup(tx, roster), TODAY)
    assert a.trades and all(t.side == "sell" for t in a.trades)  # AAPL insiders made no open-market buys
    assert a.people("buy") == 0 and a.value_sold > 0
    assert "option exercises" in a.routine_counts
    assert any(f.kind == "large_sale" for f in a.flags)
    assert all(t.role for t in a.trades)


# ---------------------------------------------------------------- institutions


def test_parse_ownership_from_captured_aapl() -> None:
    o = own.parse_ownership(load_json("yf_major_holders_aapl.json"), pd.DataFrame(load_json("yf_institutional_holders_aapl.json")))
    assert o.insider_pct == pytest.approx(0.01649, rel=1e-3)
    assert o.institution_pct == pytest.approx(0.66321)
    assert o.institution_count == 7685
    top = o.top_holders[0]
    assert top.name == "Blackrock Inc." and top.shares > 1e9 and top.pct_change is not None
    assert top.date_reported == date(2026, 6, 30)


def test_parse_ownership_handles_missing() -> None:
    o = own.parse_ownership(None, None)
    assert o.insider_pct is None and o.top_holders == []


@pytest.mark.parametrize(
    "raw, shown",
    [("COOK TIMOTHY D", "Cook, Timothy D"), ("Ternus John", "Ternus, John"), ("O'BRIEN DEIRDRE", "O'Brien, Deirdre")],
)
def test_display_name(raw: str, shown: str) -> None:
    assert own.display_name(raw) == shown
