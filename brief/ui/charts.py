"""Plotly figures for the brief. Pure functions: data in, Figure out.

Colors come from a ``Palette`` with light and dark variants matching the
"Classic research" theme. The three line colors (burgundy, ochre, navy) were
checked with a palette validator for colorblind separation, a normal-vision
floor, and 3:1 contrast against each mode's background. Bull/bear encodings use
a navy <-> burgundy diverging pair with a warm gray midpoint, never red/green,
and are always paired with labels or icons. Price and volume use separate
panels on a shared date axis rather than a dual y-axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brief.finnhub_client import AnalystConsensus, EarningsHistory
from brief.market_data import Holding
from brief.signals import BEARISH_BELOW, BULLISH_ABOVE, Signal


@dataclass(frozen=True)
class Palette:
    """Chart colors for one theme mode."""

    price: str  # categorical slot 1 (burgundy)
    sma50: str  # slot 2 (ochre)
    sma200: str  # slot 3 (navy)
    neutral: str  # volume bars, benchmark lines, "in line"
    grid: str
    text: str
    bull_strong: str
    bull: str
    hold: str
    bear: str
    bear_strong: str
    bull_tint: str  # zone backgrounds on the balance meter
    bear_tint: str
    gap: str  # thin separator between adjacent fills (matches the page background)


LIGHT = Palette(
    price="#A8323F", sma50="#B07A1E", sma200="#2E6BB0", neutral="#9C968B", grid="rgba(43,42,40,0.10)",
    text="#2B2A28", bull_strong="#24548C", bull="#7FA5D4", hold="#BDB8AE", bear="#D99AA2",
    bear_strong="#A8323F", bull_tint="rgba(46,107,176,0.10)", bear_tint="rgba(168,50,63,0.10)", gap="#FBF8F2",
)
DARK = Palette(
    price="#DB6A8A", sma50="#B88A1E", sma200="#4F8FD6", neutral="#7A746A", grid="rgba(237,230,214,0.10)",
    text="#EDE6D6", bull_strong="#5C9BE0", bull="#35618F", hold="#5C574F", bear="#8C4656",
    bear_strong="#DB6A8A", bull_tint="rgba(79,143,214,0.14)", bear_tint="rgba(219,106,138,0.14)", gap="#1C1A17",
)
FONT_FAMILY = "Inter, system-ui, sans-serif"


def _base_layout(fig: go.Figure, height: int, p: Palette) -> go.Figure:
    """Shared, recessive styling: no chart junk, light grid, compact margins."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family=FONT_FAMILY, size=12, color=p.text),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_yaxes(gridcolor=p.grid, zeroline=False, showline=False)
    return fig


def price_chart(chart: pd.DataFrame, symbol: str, p: Palette = LIGHT) -> go.Figure:
    """Close with 50/200-day SMAs above a volume panel, with range buttons."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(
        go.Scatter(x=chart.index, y=chart["Close"], name=symbol, line=dict(color=p.price, width=2),
                   hovertemplate="%{y:$,.2f}"),
        row=1, col=1,
    )
    for column, label, color in (("SMA50", "50-day avg", p.sma50), ("SMA200", "200-day avg", p.sma200)):
        if column in chart and chart[column].notna().any():
            fig.add_trace(
                go.Scatter(x=chart.index, y=chart[column], name=label, line=dict(color=color, width=1.5),
                           hovertemplate="%{y:$,.2f}"),
                row=1, col=1,
            )
    fig.add_trace(
        go.Bar(x=chart.index, y=chart["Volume"], name="Volume", marker=dict(color=p.neutral, line_width=0),
               hovertemplate="%{y:.3s}", showlegend=False),
        row=2, col=1,
    )
    _base_layout(fig, height=440, p=p)
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Volume", title_font=dict(size=11), tickformat="~s", row=2, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="1Y"),
            ],
            x=1, xanchor="right", y=1.02, yanchor="bottom",
            bgcolor="rgba(0,0,0,0)", activecolor=p.grid, font=dict(color=p.text),
        ),
        row=1, col=1,
    )
    return fig


def drawdown_chart(underwater: pd.DataFrame, p: Palette = LIGHT) -> go.Figure:
    """Percent below the running high over time; the ticker is filled, the benchmark a neutral line."""
    fig = go.Figure()
    for i, column in enumerate(underwater.columns):
        is_ticker = i == 0
        fig.add_trace(
            go.Scatter(
                x=underwater.index,
                y=underwater[column],
                name=column,
                line=dict(color=p.price if is_ticker else p.neutral, width=1.5 if is_ticker else 1.25),
                fill="tozeroy" if is_ticker else None,
                fillcolor=p.bear_tint if is_ticker else None,
                hovertemplate="%{y:.1%} below high",
            )
        )
    _base_layout(fig, height=240, p=p)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def earnings_chart(history: EarningsHistory, p: Palette = LIGHT) -> go.Figure:
    """Estimate (hollow) vs. actual (filled) EPS per quarter, oldest to newest."""
    quarters = list(reversed(history.quarters))
    labels = [f"{q.period_end:%b %Y}" for q in quarters]
    outcome_color = {"beat": p.bull_strong, "miss": p.bear_strong, "in line": p.neutral}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels, y=[q.estimate for q in quarters], name="Estimate", mode="markers",
            marker=dict(size=12, symbol="circle-open", color=p.neutral, line=dict(width=2)),
            hovertemplate="Estimate $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=[q.actual for q in quarters], name="Actual", mode="markers",
            marker=dict(size=12, color=[outcome_color.get(q.outcome or "", p.neutral) for q in quarters],
                        line=dict(width=2, color=p.gap)),
            customdata=[(q.outcome or "—").capitalize() for q in quarters],
            hovertemplate="Actual $%{y:.2f} · %{customdata}<extra></extra>",
        )
    )
    _base_layout(fig, height=220, p=p)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(tickprefix="$")
    return fig


def ratings_chart(consensus: AnalystConsensus, p: Palette = LIGHT) -> go.Figure:
    """One horizontal stacked bar of analyst ratings, Strong Buy to Strong Sell."""
    counts = {
        "Strong Buy": (consensus.strong_buy, p.bull_strong),
        "Buy": (consensus.buy, p.bull),
        "Hold": (consensus.hold, p.hold),
        "Sell": (consensus.sell, p.bear),
        "Strong Sell": (consensus.strong_sell, p.bear_strong),
    }
    fig = go.Figure()
    for label, (count, color) in counts.items():
        fig.add_trace(
            go.Bar(
                y=["Ratings"], x=[count], name=label, orientation="h",
                marker=dict(color=color, line=dict(width=2, color=p.gap)),
                text=[str(count) if count else ""], textposition="inside", insidetextanchor="middle",
                hovertemplate=f"{label}: %{{x}} analysts<extra></extra>",
            )
        )
    _base_layout(fig, height=150, p=p)
    # Legend below the bar so it can wrap on narrow screens without covering it.
    fig.update_layout(
        barmode="stack", hovermode="closest", bargap=0.2, legend_traceorder="normal",
        legend=dict(orientation="h", yanchor="top", y=-0.05, x=0),
        margin=dict(l=8, r=8, t=4, b=8),
    )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    return fig


def holdings_chart(holdings: list[Holding], p: Palette = LIGHT) -> go.Figure:
    """Horizontal bars of top ETF holdings by weight, largest at the top."""
    ordered = list(reversed(holdings))
    fig = go.Figure(
        go.Bar(
            x=[h.weight for h in ordered],
            y=[h.symbol for h in ordered],
            orientation="h",
            marker=dict(color=p.sma200, line_width=0),
            text=[f"{h.weight:.1%}" for h in ordered],
            textposition="outside",
            customdata=[h.name for h in ordered],
            hovertemplate="%{customdata}: %{x:.2%}<extra></extra>",
        )
    )
    _base_layout(fig, height=max(200, 28 * len(holdings)), p=p)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(visible=False, range=[0, max((h.weight for h in holdings), default=0.1) * 1.25])
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------- signal balance


def balance_meter(net: float, p: Palette = LIGHT) -> go.Figure:
    """A -1..+1 scale with bearish / mixed / bullish zones and a marker at the net reading."""
    fig = go.Figure()
    zones = ((-1, BEARISH_BELOW, p.bear_tint), (BEARISH_BELOW, BULLISH_ABOVE, "rgba(0,0,0,0)"), (BULLISH_ABOVE, 1, p.bull_tint))
    for x0, x1, color in zones:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, line_width=0, layer="below")
    fig.add_shape(type="line", x0=-1, x1=1, y0=0.5, y1=0.5, line=dict(color=p.neutral, width=2))
    fig.add_shape(type="line", x0=0, x1=0, y0=0.25, y1=0.75, line=dict(color=p.neutral, width=1))
    marker_color = p.bull_strong if net > BULLISH_ABOVE else p.bear_strong if net < BEARISH_BELOW else p.text
    fig.add_trace(
        go.Scatter(
            x=[net], y=[0.5], mode="markers",
            marker=dict(symbol="diamond", size=20, color=marker_color, line=dict(width=2, color=p.gap)),
            hovertemplate=f"Net reading {net:+.2f}<extra></extra>", showlegend=False,
        )
    )
    for x, text, anchor in ((-1, "Mostly bearish", "left"), (0, "Mixed", "center"), (1, "Mostly bullish", "right")):
        fig.add_annotation(x=x, y=-0.05, text=text, showarrow=False, yanchor="top", xanchor=anchor,
                           font=dict(size=12, color=p.text))
    _base_layout(fig, height=96, p=p)
    fig.update_layout(margin=dict(l=8, r=8, t=4, b=24), hovermode="closest")
    fig.update_xaxes(range=[-1.04, 1.04], visible=False)
    fig.update_yaxes(range=[-0.35, 1], visible=False)
    return fig


def signal_bars(signals: list[Signal], weights: dict[str, float], p: Palette = LIGHT) -> go.Figure:
    """Each factor's score as a diverging bar; factors weighted at 0 are faded."""
    ordered = list(reversed(signals))  # plotly draws bottom-up; keep display order top-down
    colors = [p.bull_strong if s.score > 0 else p.bear_strong if s.score < 0 else p.neutral for s in ordered]
    opacity = [1.0 if weights.get(s.key, 1.0) > 0 else 0.25 for s in ordered]
    fig = go.Figure(
        go.Bar(
            x=[s.score if s.score != 0 else 0.02 for s in ordered],  # sliver so neutral factors stay visible
            y=[s.name for s in ordered],
            orientation="h",
            marker=dict(color=colors, opacity=opacity, line_width=0),
            customdata=[[s.value, s.rule, weights.get(s.key, 1.0), s.score] for s in ordered],
            hovertemplate="<b>%{y}</b>: %{customdata[0]}<br>Score %{customdata[3]:+.2f} · weight %{customdata[2]}x"
            "<br>%{customdata[1]}<extra></extra>",
        )
    )
    _base_layout(fig, height=max(180, 30 * len(signals) + 30), p=p)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.35)
    fig.update_xaxes(range=[-1.05, 1.05], zeroline=True, zerolinecolor=p.neutral, zerolinewidth=1,
                     tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["Bearish", "", "0", "", "Bullish"], showgrid=True,
                     gridcolor=p.grid)
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig
