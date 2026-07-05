"""Streamlit UI for AI Portfolio Advisor."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_advisor import db
from portfolio_advisor.data_fetcher import (
    DEFAULT_UNIVERSE,
    compute_daily_returns,
    fetch_fundamentals,
    fetch_price_history,
)
from portfolio_advisor.holdings import (
    build_current_position_df,
    compute_rebalance_actions,
    sector_concentration,
)
from portfolio_advisor.llm_analyst import (
    active_provider,
    generate_theses_for_candidates,
    llm_available,
)
from portfolio_advisor.optimizer import optimize_portfolio, portfolio_stats
from portfolio_advisor.screener import screen_candidates

RATE_LIMIT_MAX = 5
DISCLAIMER = (
    "This tool is for research and educational purposes only. "
    "It is not licensed financial advice and does not execute trades."
)


def _load_secrets_into_env() -> None:
    """Inject Streamlit Cloud secrets into os.environ for downstream modules."""
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = st.secrets[key]


@st.cache_data(ttl=3600)
def cached_fetch_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_fundamentals(list(tickers))


@st.cache_data(ttl=3600)
def cached_fetch_price_history(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    return fetch_price_history(list(tickers), period=period)


def _init_session_state() -> None:
    if "rate_limit_session_id" not in st.session_state:
        st.session_state.rate_limit_session_id = str(uuid.uuid4())


def _rate_limit_identifier() -> str:
    headers = getattr(st.context, "headers", {}) or {}
    forwarded = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return f"session:{st.session_state.rate_limit_session_id}"


def _check_rate_limit() -> bool:
    return db.check_and_record_rate_limit(_rate_limit_identifier(), RATE_LIMIT_MAX)


def _show_rate_limit_warning() -> None:
    st.warning(
        f"Rate limit reached ({RATE_LIMIT_MAX} actions per hour). "
        "Please wait before running another analysis."
    )


def _render_llm_status() -> None:
    provider = active_provider()
    if provider == "none":
        st.warning(
            "No LLM provider configured. Set GROQ_API_KEY (recommended) or "
            "ANTHROPIC_API_KEY to enable AI thesis generation."
        )
    else:
        st.info(f"Active LLM provider: **{provider}**")


def _render_disclaimer() -> None:
    st.caption(DISCLAIMER)


def _render_past_portfolios() -> None:
    st.subheader("Past Portfolios")
    portfolios = db.list_portfolios()
    if not portfolios:
        st.write("No saved portfolios yet.")
        return

    options = {
        f"#{p['id']} — {p['created_at'][:19]} — ${p['capital']:,.0f}": p["id"]
        for p in portfolios
    }
    selected = st.selectbox("Select a portfolio", list(options.keys()))
    detail = db.get_portfolio_detail(options[selected])

    st.write(
        f"Risk: {detail['risk_tolerance']} | Style: {detail['style']} | "
        f"Horizon: {detail['horizon']}"
    )
    st.dataframe(pd.DataFrame(detail["holdings"]), use_container_width=True)

    if detail["theses"]:
        st.markdown("**Saved theses**")
        for thesis in detail["theses"]:
            with st.expander(thesis["ticker"]):
                st.write(thesis.get("thesis_summary", ""))
                st.write(f"Conviction: {thesis.get('conviction', '')}")


def _render_build_mode() -> None:
    st.header("Build New Portfolio")

    with st.sidebar:
        st.subheader("Portfolio Parameters")
        capital = st.number_input("Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
        risk_tolerance = st.selectbox(
            "Risk tolerance",
            ["conservative", "moderate", "aggressive"],
        )
        style = st.selectbox("Style tilt", ["growth", "value", "blend"])
        horizon = st.selectbox("Investment horizon", ["1y", "3y", "5y", "10y+"])
        universe_df = cached_fetch_fundamentals(tuple(DEFAULT_UNIVERSE))
        if universe_df.empty:
            st.warning("Could not load market data. Please try again later.")
        sectors = sorted(universe_df["sector"].dropna().unique().tolist()) if not universe_df.empty else []
        exclude_sectors = st.multiselect("Exclude sectors", sectors)
        max_candidates = st.slider("Max candidates", min_value=3, max_value=15, value=8)
        generate_thesis = st.checkbox("Generate AI thesis", value=False)
        build_clicked = st.button("Build Portfolio", type="primary")

    if build_clicked:
        if not _check_rate_limit():
            _show_rate_limit_warning()
            return

        if universe_df.empty:
            st.warning("Could not load market data. Please try again later.")
            return

        with st.spinner("Screening candidates..."):
            candidates = screen_candidates(
                universe_df,
                risk_tolerance=risk_tolerance,
                style=style,
                exclude_sectors=exclude_sectors,
                max_candidates=max_candidates,
            )

        if candidates.empty:
            st.warning("No candidates matched your filters. Try relaxing risk tolerance or sector exclusions.")
            return

        st.subheader("Screened Candidates")
        st.dataframe(
            candidates[
                ["ticker", "shortName", "sector", "beta", "trailingPE", "score"]
            ],
            use_container_width=True,
        )

        tickers = tuple(candidates["ticker"].tolist())
        with st.spinner("Optimizing allocation..."):
            prices = cached_fetch_price_history(tickers)
            returns = compute_daily_returns(prices)
            weights, opt_warning = optimize_portfolio(returns, risk_tolerance)
            if opt_warning:
                st.warning(opt_warning)
            stats = portfolio_stats(weights, returns)

        allocation_rows = [
            {
                "ticker": ticker,
                "weight": weight,
                "dollar_amount": capital * weight,
            }
            for ticker, weight in weights.items()
        ]
        st.subheader("Optimized Allocation")
        st.dataframe(pd.DataFrame(allocation_rows), use_container_width=True)
        st.metric("Expected annual return", f"{stats['expected_return']:.1%}")
        st.metric("Volatility", f"{stats['volatility']:.1%}")
        st.metric("Sharpe ratio", f"{stats['sharpe_ratio']:.2f}")

        theses = []
        if generate_thesis:
            if not llm_available():
                st.warning("AI thesis generation skipped — no LLM API key configured.")
            else:
                with st.spinner("Generating investment theses..."):
                    theses = generate_theses_for_candidates(
                        candidates,
                        cache_get=db.get_cached_thesis,
                        cache_set=db.save_thesis_cache,
                    )
                st.subheader("Investment Theses")
                for thesis in theses:
                    label = thesis.get("ticker", "Unknown")
                    if thesis.get("_from_cache"):
                        label += " (cached)"
                    with st.expander(label):
                        if thesis.get("error"):
                            st.error(thesis["error"])
                        else:
                            st.write(f"**Moat:** {thesis.get('moat', '')}")
                            st.write(f"**Growth drivers:** {thesis.get('growth_drivers', '')}")
                            st.write(f"**Key risks:** {thesis.get('key_risks', '')}")
                            st.write(f"**Summary:** {thesis.get('thesis_summary', '')}")
                            st.write(f"**Conviction:** {thesis.get('conviction', '')}")

        portfolio_id = db.save_portfolio(
            capital=capital,
            risk_tolerance=risk_tolerance,
            style=style,
            horizon=horizon,
            excluded_sectors=exclude_sectors,
            weights=weights,
            theses=theses,
        )
        st.success(f"Portfolio saved (ID #{portfolio_id}).")

    _render_past_portfolios()


def _render_analyze_mode() -> None:
    st.header("Analyze My Portfolio")

    default_holdings = pd.DataFrame(
        {"ticker": ["AAPL", "MSFT"], "shares": [10.0, 5.0]}
    )
    holdings_input = st.data_editor(default_holdings, num_rows="dynamic", use_container_width=True)

    analyze_action = st.radio(
        "Analysis type",
        ["Review current holdings", "Get rebalance suggestions"],
    )
    risk_tolerance = st.selectbox(
        "Risk tolerance (for rebalance target)",
        ["conservative", "moderate", "aggressive"],
        key="analyze_risk",
    )
    generate_thesis = st.checkbox("Generate AI thesis for holdings", value=False, key="analyze_thesis")
    analyze_clicked = st.button("Analyze Portfolio", type="primary")

    if analyze_clicked:
        if not _check_rate_limit():
            _show_rate_limit_warning()
            return

        rows = holdings_input.dropna(subset=["ticker"]).to_dict("records")
        rows = [r for r in rows if str(r.get("ticker", "")).strip()]
        if not rows:
            st.warning("Enter at least one ticker with shares.")
            return

        with st.spinner("Loading current positions..."):
            position_df, fundamentals_df, total_value, invalid_tickers = build_current_position_df(rows)

        if invalid_tickers:
            st.warning(
                f"Couldn't find data for: {', '.join(invalid_tickers)} — check the ticker symbols."
            )

        if position_df.empty or total_value <= 0:
            st.warning("Could not compute portfolio value from the valid tickers. Check ticker symbols and shares.")
            return

        st.subheader("Current Positions")
        st.dataframe(position_df, use_container_width=True)
        st.metric("Total portfolio value", f"${total_value:,.2f}")

        sectors = sector_concentration(position_df)
        st.subheader("Sector Concentration")
        st.bar_chart(sectors)

        if analyze_action == "Get rebalance suggestions":
            with st.spinner("Computing rebalance suggestions..."):
                actions, stats, opt_warning = compute_rebalance_actions(
                    position_df, total_value, risk_tolerance
                )
            if opt_warning:
                st.warning(opt_warning)
            st.subheader("Rebalance Suggestions")
            st.dataframe(actions, use_container_width=True)
            st.metric("Target expected return", f"{stats['expected_return']:.1%}")
            st.metric("Target volatility", f"{stats['volatility']:.1%}")
            st.metric("Target Sharpe ratio", f"{stats['sharpe_ratio']:.2f}")

        if generate_thesis:
            if not llm_available():
                st.warning("AI thesis generation skipped — no LLM API key configured.")
            else:
                with st.spinner("Generating theses for holdings..."):
                    theses = generate_theses_for_candidates(
                        fundamentals_df,
                        cache_get=db.get_cached_thesis,
                        cache_set=db.save_thesis_cache,
                    )
                st.subheader("Investment Theses")
                for thesis in theses:
                    label = thesis.get("ticker", "Unknown")
                    if thesis.get("_from_cache"):
                        label += " (cached)"
                    with st.expander(label):
                        if thesis.get("error"):
                            st.error(thesis["error"])
                        else:
                            st.write(f"**Moat:** {thesis.get('moat', '')}")
                            st.write(f"**Growth drivers:** {thesis.get('growth_drivers', '')}")
                            st.write(f"**Key risks:** {thesis.get('key_risks', '')}")
                            st.write(f"**Summary:** {thesis.get('thesis_summary', '')}")
                            st.write(f"**Conviction:** {thesis.get('conviction', '')}")


def main() -> None:
    st.set_page_config(page_title="AI Portfolio Advisor", layout="wide")
    _load_secrets_into_env()
    db.init_db()
    _init_session_state()

    st.title("AI Portfolio Advisor")
    _render_llm_status()

    mode = st.radio(
        "Mode",
        ["Build New Portfolio", "Analyze My Portfolio"],
        horizontal=True,
    )

    if mode == "Build New Portfolio":
        _render_build_mode()
    else:
        _render_analyze_mode()

    _render_disclaimer()


if __name__ == "__main__":
    main()
