"""Plotly chart helpers for Streamlit UI."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_sector_donut(sector_weights: pd.Series) -> None:
    if sector_weights.empty:
        st.write("No sector data available.")
        return
    fig = px.pie(
        values=sector_weights.values,
        names=sector_weights.index,
        title="Sector Concentration",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def render_allocation_pie(weights: dict[str, float], title: str = "Target Allocation") -> None:
    if not weights:
        return
    df = pd.DataFrame({"ticker": list(weights.keys()), "weight": list(weights.values())})
    fig = px.pie(df, values="weight", names="ticker", title=title, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def render_rebalance_bar(actions_df: pd.DataFrame) -> None:
    if actions_df.empty:
        return
    plot_df = actions_df[["ticker", "current_weight", "target_weight"]].copy()
    melted = plot_df.melt(
        id_vars="ticker",
        value_vars=["current_weight", "target_weight"],
        var_name="type",
        value_name="weight",
    )
    fig = px.bar(
        melted,
        x="weight",
        y="ticker",
        color="type",
        orientation="h",
        barmode="group",
        title="Current vs Target Weights",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_gauge(risk_profile: dict) -> None:
    value = float(risk_profile.get("gauge_value", 50))
    label = risk_profile.get("label", "Moderate")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": f"Risk Profile: {label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "#d4edda"},
                    {"range": [33, 66], "color": "#fff3cd"},
                    {"range": [66, 100], "color": "#f8d7da"},
                ],
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)
