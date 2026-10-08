"""Tests for the logo and gauge SVG builders."""

import re

import pytest

from brief.ui import charts, graphics


def test_logo_uses_theme_colors_and_current_color() -> None:
    for p in (charts.LIGHT, charts.DARK):
        svg = graphics.logo_svg(40, p)
        assert 'width="40"' in svg and p.price in svg and p.sma50 in svg
        assert p.text in svg  # first candle uses the mode's ink color (images can't inherit it)


def test_brand_sizes() -> None:
    assert "TickerBrief" in graphics.brand_html("large") and "font-size:56px" in graphics.brand_html("large")
    assert "font-size:22px" in graphics.brand_html("small")


@pytest.mark.parametrize("net, degrees", [(-1.0, -90.0), (0.0, 0.0), (0.38, 34.2), (1.0, 90.0), (2.0, 90.0)])
def test_needle_rotation(net: float, degrees: float) -> None:
    assert graphics.needle_degrees(net) == pytest.approx(degrees)


def test_dial_endpoints() -> None:
    # -1 sits at the far left of the dial, +1 at the far right, 0 straight up.
    assert graphics._point(-1) == pytest.approx((40.0, 160.0))
    assert graphics._point(1) == pytest.approx((280.0, 160.0))
    assert graphics._point(0) == pytest.approx((160.0, 40.0))


def test_gauge_markup() -> None:
    svg = graphics.gauge_svg(0.38, "Mostly bullish signals", charts.LIGHT)
    assert svg.startswith("<svg") and 'xmlns="http://www.w3.org/2000/svg"' in svg  # standalone image
    assert "rotate(34.2deg)" in svg  # needle's final position
    assert "+0.38" in svg and "Mostly bullish signals" in svg
    assert len(re.findall(r"<path ", svg)) == 3  # bearish, mixed, bullish bands
    assert "prefers-reduced-motion" in svg  # animation respects reduced-motion settings
    assert charts.LIGHT.bull_strong in svg  # center dot colored bullish


def test_gauge_escapes_label() -> None:
    assert "<b>" not in graphics.gauge_svg(0.0, "<b>x</b>")
    assert "<b>" not in graphics.gauge_html(0.0, "<b>x</b>")


def test_gauge_html_embeds_image_and_readout() -> None:
    html = graphics.gauge_html(-0.5, "Mostly bearish signals", charts.DARK)
    assert html.count('<img src="data:image/svg+xml;base64,') == 1
    assert "-0.50" in html and "Mostly bearish signals" in html


def test_brand_embeds_logo_as_image() -> None:
    assert '<img src="data:image/svg+xml;base64,' in graphics.brand_html("small")
