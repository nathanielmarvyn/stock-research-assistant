"""Shared fixtures: real yfinance payloads captured to tests/fixtures/."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> Any:
    """Load a JSON fixture by file name."""
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def aapl_info() -> dict:
    return load_json("info_aapl.json")


@pytest.fixture
def spy_info() -> dict:
    return load_json("info_spy.json")


@pytest.fixture
def unknown_info() -> dict:
    return load_json("info_zzzzq.json")


@pytest.fixture
def spy_holdings_df() -> pd.DataFrame:
    """SPY top holdings in yfinance's shape: index=Symbol, 'Holding Percent' column."""
    records = load_json("spy_top_holdings.json")
    df = pd.DataFrame(records).rename(columns={"weight": "Holding Percent"})
    return df.set_index("Symbol")
