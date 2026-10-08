"""Risk metric tests against hand-calculated examples."""

import math

import pandas as pd
import pytest

from brief import risk


def series(values, start="2026-01-05", freq="B") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


# ---------------------------------------------------------------- volatility, beta, correlation


def test_annualized_volatility_hand_calculated() -> None:
    # Returns +1%, -1%, +1%, -1%: mean 0, sample std = sqrt(4 * 0.0001 / 3) = 0.011547
    r = series([0.01, -0.01, 0.01, -0.01])
    assert risk.annualized_volatility(r) == pytest.approx(math.sqrt(4 * 0.0001 / 3) * math.sqrt(252))


def test_beta_and_correlation_of_leveraged_series() -> None:
    bench = series([0.01, -0.02, 0.015, 0.005, -0.01])
    asset = bench * 2  # moves exactly twice as much
    assert risk.beta(asset, bench) == pytest.approx(2.0)
    assert risk.correlation(asset, bench) == pytest.approx(1.0)


def test_beta_uses_common_dates_only() -> None:
    bench = series([0.01, -0.02, 0.015, 0.005])
    asset = (bench * 1.5).iloc[1:]  # missing the first day
    assert risk.beta(asset, bench) == pytest.approx(1.5)


def test_beta_undefined_for_flat_benchmark() -> None:
    assert risk.beta(series([0.01, 0.02, 0.03]), series([0.0, 0.0, 0.0])) is None


# ---------------------------------------------------------------- drawdown


def test_max_drawdown_dates_and_recovery() -> None:
    # 120 -> 90 is -25% (the worst); 130 -> 104 is only -20%
    close = series([100, 120, 90, 95, 130, 104])
    dd = risk.max_drawdown(close)
    assert dd.depth == pytest.approx(-0.25)
    assert dd.peak_date == close.index[1] and dd.trough_date == close.index[2]
    assert dd.recovery_date == close.index[4]  # first close back at or above 120


def test_max_drawdown_not_recovered() -> None:
    dd = risk.max_drawdown(series([100, 80, 90]))
    assert dd.depth == pytest.approx(-0.2) and dd.recovery_date is None


def test_no_drawdown_on_rising_series() -> None:
    assert risk.max_drawdown(series([1, 2, 3])).depth == 0


def test_drawdown_series() -> None:
    assert list(risk.drawdown_series(series([100, 120, 90]))) == pytest.approx([0, 0, -0.25])


# ---------------------------------------------------------------- Sharpe


def test_sharpe_hand_calculated() -> None:
    assert risk.sharpe_ratio(0.12, 0.04, 0.20) == pytest.approx(0.4)


@pytest.mark.parametrize("args", [(None, 0.04, 0.2), (0.1, None, 0.2), (0.1, 0.04, 0.0)])
def test_sharpe_undefined(args) -> None:
    assert risk.sharpe_ratio(*args) is None


# ---------------------------------------------------------------- capture


def test_capture_ratios_hand_calculated(monkeypatch) -> None:
    monkeypatch.setattr(risk, "MIN_CAPTURE_MONTHS", 4)
    asset = series([0.02, -0.01, 0.04, -0.02], freq="ME")
    bench = series([0.01, -0.02, 0.02, -0.01], freq="ME")
    up, down = risk.capture_ratios(asset, bench)
    # Up months (bench > 0): 1 and 3. Down months: 2 and 4. Geometric means:
    expected_up = (math.sqrt(1.02 * 1.04) - 1) / (math.sqrt(1.01 * 1.02) - 1)
    expected_down = (math.sqrt(0.99 * 0.98) - 1) / (math.sqrt(0.98 * 0.99) - 1)
    assert up == pytest.approx(expected_up)  # ~2.0: gains about twice the market's in up months
    assert down == pytest.approx(expected_down)  # exactly 1.0: same compounded loss in down months


def test_capture_needs_enough_months() -> None:
    m = series([0.01] * 6, freq="ME")
    assert risk.capture_ratios(m, m) == (None, None)


def test_monthly_returns_drop_month_in_progress() -> None:
    close = series(range(100, 100 + 70), start="2026-01-01", freq="B")  # ends mid-April
    months = risk.monthly_returns(close)
    assert months.index[-1].month == 3  # April is incomplete, so it's excluded


# ---------------------------------------------------------------- section


def history(closes) -> pd.DataFrame:
    idx = pd.bdate_range("2024-10-01", periods=len(closes), tz="America/New_York")
    return pd.DataFrame({"Close": closes, "Volume": [1e6] * len(closes)}, index=idx)


def test_profile_against_benchmark() -> None:
    bench = [100 * (1 + 0.01 * math.sin(i / 5)) * 1.0003**i for i in range(504)]
    asset = [100 * (1 + 0.02 * math.sin(i / 5)) * 1.0006**i for i in range(504)]
    p = risk.build_risk_profile(history(asset), history(bench), "SPY", 0.04, "XYZ")
    assert p.ticker.volatility_1y > p.benchmark.volatility_1y  # twice the swings
    assert p.ticker.beta_2y > 1.5 and p.ticker.correlation_2y > 0.9
    assert p.benchmark.beta_2y == pytest.approx(1.0)
    assert list(p.underwater.columns) == ["XYZ", "SPY"]
    assert p.capture_months >= 20


def test_profile_for_benchmark_itself_has_no_comparison() -> None:
    closes = [100 * 1.0004**i for i in range(504)]
    p = risk.build_risk_profile(history(closes), history(closes), "SPY", 0.04, "SPY")
    assert p.benchmark is None and p.ticker.beta_2y is None
    assert list(p.underwater.columns) == ["SPY"]


def test_missing_risk_free_only_blanks_sharpe() -> None:
    closes = [100 * 1.0004**i for i in range(504)]
    p = risk.build_risk_profile(history(closes), None, "SPY", None, "XYZ")
    assert p.ticker.sharpe_1y is None and p.ticker.volatility_1y is not None


def test_short_history_rejected() -> None:
    with pytest.raises(risk.DataUnavailableError):
        risk.build_risk_profile(history([100.0] * 30), None, "SPY", 0.04, "NEW")


def metrics(vol, sharpe, up=None, down=None) -> risk.RiskMetrics:
    return risk.RiskMetrics(vol, 0.1, sharpe, None, None, 1.0, 0.5, up, down)


def test_risk_summary_more_volatile_with_capture() -> None:
    p = risk.RiskProfile(metrics(0.26, 1.17, 1.0, 0.55), metrics(0.13, 0.96), "SPY", 0.04, 23, pd.DataFrame())
    assert risk.risk_summary(p, "AAPL") == (
        "AAPL has been about 2.0x as volatile as SPY over the past year, with a better risk-adjusted "
        "return (Sharpe 1.17 vs. 0.96). In the market's down months it fell 55% as much as SPY, "
        "and in up months it gained 100% as much."
    )


def test_risk_summary_calmer_and_none_without_benchmark() -> None:
    p = risk.RiskProfile(metrics(0.10, 0.5), metrics(0.13, 0.96), "SPY", 0.04, 0, pd.DataFrame())
    assert risk.risk_summary(p, "KO").startswith("KO has been calmer than SPY, at about 0.8x")
    assert risk.risk_summary(risk.RiskProfile(metrics(0.1, 0.5), None, "SPY", 0.04, 0, pd.DataFrame()), "SPY") is None
