"""Hand-built SVG graphics: the TickerBrief logo and the signal-balance gauge.

Pure functions returning markup strings, so they're easy to test and theme.
Colors come from the chart ``Palette`` (light or dark).

The SVGs are delivered as data-URI <img> tags because Streamlit's st.html
sanitizer strips inline <svg>. Inside an image the SVG can't inherit the page's
text color, so "ink" comes from the palette; CSS animations defined inside the
SVG still run.
"""

from __future__ import annotations

import base64
import math
from html import escape

from brief.signals import BEARISH_BELOW, BULLISH_ABOVE
from brief.ui.charts import LIGHT, Palette

APP_NAME = "TickerBrief"


def svg_img(svg: str, width: str, alt: str) -> str:
    """Wrap an SVG document as an <img> (st.html keeps images but strips inline SVG)."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{data}" alt="{escape(alt)}" style="width:{width};height:auto;display:block">'


def logo_svg(size: int = 44, p: Palette = LIGHT) -> str:
    """Three rising candlesticks: ink, ochre, burgundy (the chart colors)."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 44 44" role="img" aria-label="{APP_NAME} logo" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="11" y1="22" x2="11" y2="38" stroke="{p.text}" stroke-width="1.6"/>'
        f'<rect x="7" y="26" width="8" height="10" rx="1.5" fill="{p.text}"/>'
        f'<line x1="22" y1="14" x2="22" y2="34" stroke="{p.sma50}" stroke-width="1.6"/>'
        f'<rect x="18" y="18" width="8" height="12" rx="1.5" fill="{p.sma50}"/>'
        f'<line x1="33" y1="5" x2="33" y2="28" stroke="{p.price}" stroke-width="1.6"/>'
        f'<rect x="29" y="8" width="8" height="15" rx="1.5" fill="{p.price}"/>'
        "</svg>"
    )


def brand_html(size: str = "large", p: Palette = LIGHT, href: str | None = None) -> str:
    """Logo plus the TickerBrief wordmark in the heading serif; a link when ``href`` is given."""
    logo_px, text_px, gap = (64, 56, 16) if size == "large" else (30, 22, 8)
    brand = (
        f'<div class="tb-brand tb-brand-{size}" style="display:flex;align-items:center;gap:{gap}px;'
        f'justify-content:{"center" if size == "large" else "flex-start"}">'
        f"{svg_img(logo_svg(logo_px, p), f'{logo_px}px', APP_NAME + ' logo')}"
        f'<span style="font-family:Lora,Georgia,serif;font-weight:600;font-size:{text_px}px;'
        f'letter-spacing:-0.01em;line-height:1">{APP_NAME}</span></div>'
    )
    if href is None:
        return brand
    return (
        f'<a class="tb-home-link" href="{escape(href)}" title="Back to home" '
        f'aria-label="{APP_NAME} home">{brand}</a>'
    )


# ---------------------------------------------------------------- gauge

_CX, _CY, _R, _STROKE = 160.0, 160.0, 120.0, 22.0


def _point(value: float, radius: float = _R) -> tuple[float, float]:
    """Position on the half-dial for a value in [-1, 1] (-1 = far left, +1 = far right)."""
    angle = math.pi * (1 - (value + 1) / 2)  # 180 deg at -1, 0 deg at +1
    return _CX + radius * math.cos(angle), _CY - radius * math.sin(angle)


def _arc(start: float, end: float, color: str, opacity: float = 1.0) -> str:
    """One colored band of the dial between two values."""
    (x0, y0), (x1, y1) = _point(start), _point(end)
    large = 1 if (end - start) > 1 else 0
    return (
        f'<path d="M{x0:.1f} {y0:.1f} A{_R:.0f} {_R:.0f} 0 {large} 1 {x1:.1f} {y1:.1f}" fill="none" '
        f'stroke="{color}" stroke-width="{_STROKE:.0f}" stroke-opacity="{opacity}" stroke-linecap="butt"/>'
    )


def needle_degrees(net: float) -> float:
    """Needle rotation from straight up: -90 deg at -1 (bearish), +90 deg at +1 (bullish)."""
    return max(-1.0, min(1.0, net)) * 90.0


def gauge_svg(net: float, label: str, p: Palette = LIGHT) -> str:
    """Half-dial SVG: bearish / mixed / bullish bands and a needle that sweeps into place."""
    gap = 0.012  # thin separator between bands
    bands = (
        _arc(-1, BEARISH_BELOW - gap, p.bear_strong, 0.85)
        + _arc(BEARISH_BELOW + gap, BULLISH_ABOVE - gap, p.hold)
        + _arc(BULLISH_ABOVE + gap, 1, p.bull_strong, 0.85)
    )
    inner, outer = _R - _STROKE / 2 - 3, _R - _STROKE / 2 - 10
    ticks = "".join(
        f'<line x1="{_point(v, inner)[0]:.1f}" y1="{_point(v, inner)[1]:.1f}" '
        f'x2="{_point(v, outer)[0]:.1f}" y2="{_point(v, outer)[1]:.1f}" '
        f'stroke="{p.text}" stroke-opacity="0.35" stroke-width="1.5"/>'
        for v in (-1, -0.5, 0, 0.5, 1)
    )
    deg = needle_degrees(net)
    needle_len = _R - _STROKE / 2 - 14
    hub = p.bull_strong if net > BULLISH_ABOVE else p.bear_strong if net < BEARISH_BELOW else p.text
    label_style = f'font-family="Inter,Arial,sans-serif" font-size="13" fill="{p.text}" fill-opacity="0.75"'
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 196" role="img" '
        f'aria-label="Signal balance: {escape(label)}, net {net:+.2f} from -1 bearish to +1 bullish">'
        "<style>"
        f"@keyframes sweep {{ from {{ transform: rotate(-90deg); }} to {{ transform: rotate({deg:.1f}deg); }} }}"
        f".needle {{ transform-origin: {_CX:.0f}px {_CY:.0f}px; transform: rotate({deg:.1f}deg); "
        "animation: sweep 0.9s cubic-bezier(.2,.8,.2,1) both; }"
        "@media (prefers-reduced-motion: reduce) { .needle { animation: none; } }"
        "</style>"
        f"{bands}{ticks}"
        f'<g class="needle"><line x1="{_CX:.0f}" y1="{_CY:.0f}" x2="{_CX:.0f}" y2="{_CY - needle_len:.0f}" '
        f'stroke="{p.text}" stroke-width="4" stroke-linecap="round"/></g>'
        f'<circle cx="{_CX:.0f}" cy="{_CY:.0f}" r="9" fill="{p.text}"/>'
        f'<circle cx="{_CX:.0f}" cy="{_CY:.0f}" r="4" fill="{hub}"/>'
        f'<text x="{_point(-1)[0]:.0f}" y="188" text-anchor="middle" {label_style}>Bearish</text>'
        f'<text x="{_CX:.0f}" y="{_CY - _R - _STROKE / 2 - 8:.0f}" text-anchor="middle" {label_style}>Mixed</text>'
        f'<text x="{_point(1)[0]:.0f}" y="188" text-anchor="middle" {label_style}>Bullish</text>'
        "</svg>"
    )


def gauge_html(net: float, label: str, p: Palette = LIGHT) -> str:
    """The gauge image with the net reading and its label underneath."""
    image = svg_img(gauge_svg(net, label, p), "100%", f"Signal balance gauge: {label}, net {net:+.2f}")
    return (
        '<div class="tb-gauge" style="max-width:420px;margin:0 auto">'
        f"{image}"
        '<div style="text-align:center;margin-top:-4px">'
        f'<div style="font-family:Lora,Georgia,serif;font-size:28px;font-weight:600;line-height:1.2">{net:+.2f}</div>'
        f'<div style="font-size:15px;opacity:0.85">{escape(label)}</div></div></div>'
    )
