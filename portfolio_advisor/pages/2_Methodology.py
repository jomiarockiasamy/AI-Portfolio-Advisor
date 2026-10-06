"""Methodology and metrics glossary reference page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portfolio_advisor.rebalance_engine import (
    BUY_MIN_WEIGHT,
    REBALANCE_MODE_DESCRIPTIONS,
    REBALANCE_MODE_LABELS,
)
from portfolio_advisor.signal_engine import (
    BUY_THRESHOLD,
    COMPONENT_DESCRIPTIONS,
    COMPONENT_LABELS,
    FUND_SCORING_NOTE,
    SELL_THRESHOLD,
    WEIGHTS,
)
from portfolio_advisor.ui_bootstrap import bootstrap_app, render_disclaimer

bootstrap_app(page_title="Methodology — AI Portfolio Advisor")

st.title("Methodology & Glossary")
st.caption(
    "Scoring weights and thresholds are loaded live from the signal engine — "
    "they always match the per-ticker breakdown in thesis expanders."
)

tab_scoring, tab_glossary = st.tabs(["Scoring Methodology", "Metrics Glossary"])

with tab_scoring:
    st.subheader("Composite signal formula")
    formula_parts = [f"{COMPONENT_LABELS[name]} × {weight:.0%}" for name, weight in WEIGHTS.items()]
    st.markdown("**Composite score** = " + " + ".join(formula_parts))
    st.markdown(
        f"- **BUY** when composite ≥ **{BUY_THRESHOLD:.0f}**\n"
        f"- **HOLD** when composite is **{SELL_THRESHOLD + 1:.0f}–{BUY_THRESHOLD - 1:.0f}**\n"
        f"- **SELL** when composite ≤ **{SELL_THRESHOLD:.0f}**"
    )
    st.caption(
        "Conviction (high / medium / low) is derived from these composite bands — "
        "it is not an independent LLM judgment."
    )

    rows = []
    for name, weight in WEIGHTS.items():
        rows.append(
            {
                "Component": COMPONENT_LABELS[name],
                "Weight": f"{weight:.0%}",
                "What it measures": COMPONENT_DESCRIPTIONS[name],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Funds and ETFs")
    st.write(FUND_SCORING_NOTE)
    st.write(
        "Thesis generation uses a separate fund branch: the LLM writes **fund_role** "
        "(index exposure, cost, portfolio role) instead of corporate **moat** language."
    )

    st.subheader("Rebalance modes")
    st.markdown(
        "In **Analyze → Get rebalance suggestions**, choose how target weights are computed:"
    )
    for mode_key, label in REBALANCE_MODE_LABELS.items():
        st.markdown(f"- **{label}**: {REBALANCE_MODE_DESCRIPTIONS[mode_key]}")
    st.markdown(
        f"Signal-aligned tilt rules (thresholds from live engine: BUY ≥ {BUY_THRESHOLD:.0f}, "
        f"SELL ≤ {SELL_THRESHOLD:.0f}):\n"
        f"- **BUY**: multiply optimizer weight, floor at **{BUY_MIN_WEIGHT:.0%}** minimum\n"
        f"- **HOLD**: floor at **50% of current weight** (prevents zeroing core ETFs)\n"
        f"- **SELL**: trim only, no floor\n"
        f"- Weights are capped per risk tolerance, then iteratively rebalanced to sum to 100% "
        f"without exceeding the cap."
    )

with tab_glossary:
    glossary_rows = [
        {
            "Term": "Trailing P/E",
            "What it is": "Price ÷ last 12 months' EPS",
            "How to read it": "Lower = cheaper vs past profit",
        },
        {
            "Term": "Forward P/E",
            "What it is": "Price ÷ estimated next-12m EPS",
            "How to read it": "Below trailing → expected growth",
        },
        {
            "Term": "PEG ratio",
            "What it is": "P/E ÷ earnings growth rate",
            "How to read it": "~1 fair; >2 often unreliable when growth denominator is tiny",
        },
        {
            "Term": "Profit margin",
            "What it is": "Net income ÷ revenue",
            "How to read it": "0.21 = 21% of revenue is profit",
        },
        {
            "Term": "Revenue growth",
            "What it is": "YoY top-line growth",
            "How to read it": "Straightforward momentum in sales",
        },
        {
            "Term": "Earnings growth",
            "What it is": "YoY profit growth",
            "How to read it": "Can inflate off a low base — check reliability",
        },
        {
            "Term": "Dividend yield",
            "What it is": "Annual dividend ÷ price",
            "How to read it": (
                "yfinance scale varies. In this app, values like 0.43 often mean "
                "0.43% already, not 0.0043. The formatter detects scale before display."
            ),
        },
        {
            "Term": "Total assets (funds)",
            "What it is": "Fund AUM",
            "How to read it": "Scale and liquidity, not quality by itself",
        },
        {
            "Term": "Debt-to-equity",
            "What it is": "Debt ÷ equity",
            "How to read it": "yfinance may use percentage scale (600 ≈ 6.0×) — verify before trusting",
        },
        {
            "Term": "Beta",
            "What it is": "Volatility vs market (1.0 = market)",
            "How to read it": "2.5 ≈ 2.5× market swings",
        },
    ]
    st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True, hide_index=True)

render_disclaimer()
