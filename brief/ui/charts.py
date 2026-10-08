"""Plotly figures for the brief. Pure functions: data in, Figure out.

Color roles follow a colorblind-validated palette: categorical slots 1-3 for the
price lines, a blue<->red diverging scale (gray midpoint) for analyst ratings,
and a neutral gray for volume. Price and volume use separate panels on a shared
date axis rather than a dual y-axis.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brief.finnhub_client import AnalystConsensus, EarningsHistory
from brief.market_data import Holding

PRICE, SMA50, SMA200 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
VOLUME = "#9a9993"
RATING_COLORS = {  # diverging: buy (blue) <-> sell (red), neutral gray midpoint
    "Strong Buy": "#1c5cab",
    "Buy": "#6da7ec",
    "Hold": "#b5b4ae",
    "Sell": "#ec8a89",
    "Strong Sell": "#c22f2f",
}
FONT = dict(family="Inter, system-ui, sans-serif", size=12)


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    """Shared, recessive styling: no chart junk, light grid, compact margins."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        font=FONT,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False, showline=False)
    return fig


def price_chart(chart: pd.DataFrame, symbol: str) -> go.Figure:
    """Close with 50/200-day SMAs above a volume panel, with range buttons."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(
        go.Scatter(x=chart.index, y=chart["Close"], name=symbol, line=dict(color=PRICE, width=2),
                   hovertemplate="%{y:$,.2f}"),
        row=1, col=1,
    )
    for column, label, color in (("SMA50", "50-day avg", SMA50), ("SMA200", "200-day avg", SMA200)):
        if column in chart and chart[column].notna().any():
            fig.add_trace(
                go.Scatter(x=chart.index, y=chart[column], name=label, line=dict(color=color, width=1.5),
                           hovertemplate="%{y:$,.2f}"),
                row=1, col=1,
            )
    fig.add_trace(
        go.Bar(x=chart.index, y=chart["Volume"], name="Volume", marker=dict(color=VOLUME, line_width=0),
               hovertemplate="%{y:.3s}", showlegend=False),
        row=2, col=1,
    )
    _base_layout(fig, height=440)
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
        ),
        row=1, col=1,
    )
    return fig


def drawdown_chart(underwater: pd.DataFrame) -> go.Figure:
    """Percent below the running high over time; the ticker is filled, the benchmark a gray line."""
    fig = go.Figure()
    for i, column in enumerate(underwater.columns):
        is_ticker = i == 0
        fig.add_trace(
            go.Scatter(
                x=underwater.index,
                y=underwater[column],
                name=column,
                line=dict(color=PRICE if is_ticker else VOLUME, width=1.5 if is_ticker else 1.25),
                fill="tozeroy" if is_ticker else None,
                fillcolor="rgba(42,120,214,0.12)" if is_ticker else None,
                hovertemplate="%{y:.1%} below high",
            )
        )
    _base_layout(fig, height=240)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def earnings_chart(history: EarningsHistory) -> go.Figure:
    """Estimate (hollow) vs. actual (filled) EPS per quarter, oldest to newest."""
    quarters = list(reversed(history.quarters))
    labels = [f"{q.period_end:%b %Y}" for q in quarters]
    outcome_color = {"beat": RATING_COLORS["Strong Buy"], "miss": RATING_COLORS["Strong Sell"], "in line": VOLUME}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels, y=[q.estimate for q in quarters], name="Estimate", mode="markers",
            marker=dict(size=12, symbol="circle-open", color=VOLUME, line=dict(width=2)),
            hovertemplate="Estimate $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=[q.actual for q in quarters], name="Actual", mode="markers",
            marker=dict(size=12, color=[outcome_color.get(q.outcome or "", VOLUME) for q in quarters],
                        line=dict(width=2, color="rgba(255,255,255,0.9)")),
            customdata=[(q.outcome or "—").capitalize() for q in quarters],
            hovertemplate="Actual $%{y:.2f} · %{customdata}<extra></extra>",
        )
    )
    _base_layout(fig, height=220)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(tickprefix="$")
    return fig


def ratings_chart(consensus: AnalystConsensus) -> go.Figure:
    """One horizontal stacked bar of analyst ratings, Strong Buy to Strong Sell."""
    counts = {
        "Strong Buy": consensus.strong_buy,
        "Buy": consensus.buy,
        "Hold": consensus.hold,
        "Sell": consensus.sell,
        "Strong Sell": consensus.strong_sell,
    }
    fig = go.Figure()
    for label, count in counts.items():
        fig.add_trace(
            go.Bar(
                y=["Ratings"], x=[count], name=label, orientation="h",
                marker=dict(color=RATING_COLORS[label], line=dict(width=2, color="rgba(255,255,255,0.9)")),
                text=[str(count) if count else ""], textposition="inside", insidetextanchor="middle",
                hovertemplate=f"{label}: %{{x}} analysts<extra></extra>",
            )
        )
    _base_layout(fig, height=120)
    fig.update_layout(barmode="stack", hovermode="closest", bargap=0.2, legend_traceorder="normal")
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    return fig


def holdings_chart(holdings: list[Holding]) -> go.Figure:
    """Horizontal bars of top ETF holdings by weight, largest at the top."""
    ordered = list(reversed(holdings))
    fig = go.Figure(
        go.Bar(
            x=[h.weight for h in ordered],
            y=[h.symbol for h in ordered],
            orientation="h",
            marker=dict(color=PRICE, line_width=0),
            text=[f"{h.weight:.1%}" for h in ordered],
            textposition="outside",
            customdata=[h.name for h in ordered],
            hovertemplate="%{customdata}: %{x:.2%}<extra></extra>",
        )
    )
    _base_layout(fig, height=max(200, 28 * len(holdings)))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(visible=False, range=[0, max((h.weight for h in holdings), default=0.1) * 1.25])
    fig.update_yaxes(showgrid=False)
    return fig
