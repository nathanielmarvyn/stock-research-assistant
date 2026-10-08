"""Number formatting shared by the UI, the preview script, and the AI prompt.

Feeding the model pre-formatted strings ("$109.42B", "+16.4%") instead of raw
floats (109417000000.0, 0.164) removes a common source of unit mistakes.
"""

from __future__ import annotations

MISSING = "—"


def money(value: float | None, decimals: int = 2) -> str:
    """$1,234.56 for prices; $1.23T / $4.56B / $7.89M for large amounts."""
    if value is None:
        return MISSING
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"{'-' if value < 0 else ''}${abs(value) / size:,.{decimals}f}{suffix}"
    return f"{'-' if value < 0 else ''}${abs(value):,.{decimals}f}"


def pct(value: float | None, decimals: int = 1, signed: bool = False) -> str:
    """Fraction as a percent: 0.164 -> '16.4%' (or '+16.4%' when signed)."""
    if value is None:
        return MISSING
    return f"{value:{'+' if signed else ''}.{decimals}%}"


def num(value: float | None, decimals: int = 2, suffix: str = "") -> str:
    """Plain number with thousands separators: 38.26 -> '38.26', 1.5 -> '1.50x' with suffix='x'."""
    if value is None:
        return MISSING
    return f"{value:,.{decimals}f}{suffix}"


def volume(value: float | None) -> str:
    """Share volume: 34128200 -> '34.1M'."""
    if value is None:
        return MISSING
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:,.1f}{suffix}"
    return f"{value:,.0f}"


def money_compact(value: float | None) -> str:
    """Dollar amounts for trade sizes: $43.4M, $519K, $850 (unlike money(), abbreviates thousands)."""
    if value is None:
        return MISSING
    if abs(value) >= 1e6:
        return money(value, 1)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e3:,.0f}K" if abs(value) >= 1e3 else f"{sign}${abs(value):,.0f}"
