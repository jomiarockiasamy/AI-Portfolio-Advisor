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
from portfolio_advisor.charts import (
    render_allocation_pie,
    render_rebalance_bar,
    render_risk_gauge,
    render_sector_donut,
)
from portfolio_advisor.data_fetcher import (
    DEFAULT_UNIVERSE,
    compute_daily_returns,
    compute_momentum_scores,
    fetch_fundamentals,
    fetch_price_history,
    is_fund,
)
from portfolio_advisor.holdings import (
    build_current_position_df,
    compute_rebalance_actions,
    enrich_with_finnhub,
    sector_concentration,
)
from portfolio_advisor.llm_analyst import (
    NO_NEWS_FALLBACK,
    active_provider,
    build_relative_context,
    generate_theses_for_candidates,
    generate_thesis,
    llm_available,
    score_news_sentiments_for_tickers,
    thesis_is_complete,
    _thesis_narrative_empty,
)
from portfolio_advisor.optimizer import RISK_PARAMS, optimize_portfolio, portfolio_stats
from portfolio_advisor.rebalance_engine import REBALANCE_MODE_DESCRIPTIONS, REBALANCE_MODE_LABELS
from portfolio_advisor.risk_profile import classify_risk_level
from portfolio_advisor.signal_engine import (
    COMPONENT_LABELS,
    FUND_SCORING_NOTE,
    WEIGHTS,
    build_signals_for_holdings,
)
from portfolio_advisor.screener import find_sector_alternatives, screen_candidates
from portfolio_advisor.ui_bootstrap import DISCLAIMER, bootstrap_app, render_disclaimer

RATE_LIMIT_MAX = 5
RATE_LIMIT_MAX_DEV = 100


def _load_dotenv() -> None:
    from portfolio_advisor.ui_bootstrap import load_dotenv

    load_dotenv()


def _load_secrets_into_env() -> None:
    from portfolio_advisor.ui_bootstrap import load_secrets_into_env

    load_secrets_into_env()


@st.cache_data(ttl=3600)
def cached_fetch_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_fundamentals(list(tickers))


@st.cache_data(ttl=3600)
def cached_fetch_price_history(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    return fetch_price_history(list(tickers), period=period)


def _dev_mode_enabled() -> bool:
    return os.environ.get("DEV_MODE", "").lower().strip() in {"1", "true", "yes"}


def _rate_limit_max() -> int:
    if _dev_mode_enabled():
        return RATE_LIMIT_MAX_DEV
    return RATE_LIMIT_MAX


def _init_session_state() -> None:
    if "rate_limit_session_id" not in st.session_state:
        st.session_state.rate_limit_session_id = str(uuid.uuid4())


def _rate_limit_identifier(action: str) -> str:
    headers = getattr(st.context, "headers", {}) or {}
    forwarded = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
    if forwarded:
        base = forwarded.split(",")[0].strip()
    else:
        base = f"session:{st.session_state.rate_limit_session_id}"
    return f"{base}:{action}"


def _check_rate_limit(action: str) -> bool:
    if _dev_mode_enabled() and os.environ.get("DEV_MODE_DISABLE_RATE_LIMIT", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    return db.check_and_record_rate_limit(_rate_limit_identifier(action), _rate_limit_max())


def _show_rate_limit_warning() -> None:
    st.warning(
        f"Rate limit reached ({_rate_limit_max()} actions per hour for this mode). "
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
    render_disclaimer()


def _signal_badge_color(signal: str) -> str:
    return {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(signal, "gray")


def _short_name_lookup(fundamentals_df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, row in fundamentals_df.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if ticker:
            lookup[ticker] = str(row.get("shortName") or ticker)
    return lookup


def _thesis_expander_label(
    ticker: str,
    short_name: str | None,
    thesis: dict,
    *,
    enriched: dict | None = None,
    is_fund_flag: bool = False,
) -> str:
    name = short_name or ticker
    base = f"{ticker} — {name}"
    failed = bool(thesis.get("error")) or not thesis_is_complete(
        thesis,
        is_fund=is_fund_flag,
        enriched_context=(enriched or {}).get(ticker),
    )
    if failed:
        return f"{base} — failed"
    if thesis.get("_from_cache"):
        return f"{base} (cached)"
    return base


def _render_score_breakdown(signal: dict, *, is_fund_flag: bool = False) -> None:
    components = signal.get("components") or []
    if not components:
        return

    rows = []
    for item in components:
        key = str(item.get("component", ""))
        label = COMPONENT_LABELS.get(key, key)
        weight = WEIGHTS.get(key, item.get("weight", 0))
        raw = item.get("score", 0)
        contribution = item.get("contribution", raw * weight)
        rows.append(
            {
                "Component": label,
                "Weight": f"{weight:.0%}",
                "Raw score": f"{raw}/100",
                "Weighted contribution": round(float(contribution), 1),
            }
        )

    composite = signal.get("composite_score", 0)
    signal_label = signal.get("signal", "HOLD")
    rows.append(
        {
            "Component": "**Composite**",
            "Weight": "",
            "Raw score": "",
            "Weighted contribution": f"**{composite} → {signal_label}**",
        }
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "See **Methodology** in the sidebar for what these numbers mean and how scores are calculated."
    )
    if is_fund_flag:
        st.caption(FUND_SCORING_NOTE)


def _render_thesis_expander(
    thesis: dict,
    signal: dict | None = None,
    *,
    is_fund_flag: bool = False,
) -> None:
    if signal:
        color = _signal_badge_color(signal.get("signal", "HOLD"))
        st.markdown(
            f":{color}[**{signal.get('signal', 'HOLD')}**] "
            f"Signal strength: **{signal.get('composite_score', '—')}**/100"
        )
        if signal.get("conviction"):
            st.caption(f"Conviction: **{signal.get('conviction')}** (derived from signal score)")
        _render_score_breakdown(signal, is_fund_flag=is_fund_flag)
        if signal.get("news_insufficient_data"):
            st.caption("News sentiment: insufficient news data (neutral).")

    if thesis.get("error"):
        st.error(f"{thesis.get('ticker', 'Unknown')}: {thesis['error']}")
        return

    if not thesis_is_complete(thesis, is_fund=is_fund_flag) or _thesis_narrative_empty(thesis):
        st.error(
            "Thesis generation failed — check terminal logs or use Retry failed theses."
        )
        return

    cited = thesis.get("cited_inputs") or []
    if cited:
        st.markdown("**Cited inputs**")
        for item in cited:
            st.markdown(f"- {item}")
        st.caption(
            "See **Methodology** in the sidebar for what these numbers mean and how scores are calculated."
        )

    if thesis.get("signal_reasoning"):
        st.write(f"**Signal reasoning:** {thesis.get('signal_reasoning')}")

    if thesis.get("no_news_note"):
        note = str(thesis.get("no_news_note"))
        if note in {NO_NEWS_FALLBACK, "{NO_NEWS_FALLBACK}"}:
            note = NO_NEWS_FALLBACK
        st.caption(note)

    if thesis.get("vs_alternatives"):
        st.write(f"**Vs alternatives:** {thesis.get('vs_alternatives')}")

    if thesis.get("fund_role"):
        st.write(f"**Fund role:** {thesis.get('fund_role')}")
    elif thesis.get("moat"):
        st.write(f"**Moat:** {thesis.get('moat', '')}")

    st.write(f"**Growth drivers:** {thesis.get('growth_drivers', '')}")
    st.write(f"**Key risks:** {thesis.get('key_risks', '')}")
    st.write(f"**Summary:** {thesis.get('thesis_summary', '')}")


def _thesis_failed(
    thesis: dict,
    *,
    enriched: dict | None = None,
    is_fund_flags: dict[str, bool] | None = None,
) -> bool:
    ticker = str(thesis.get("ticker", "")).upper()
    return bool(thesis.get("error")) or not thesis_is_complete(
        thesis,
        is_fund=(is_fund_flags or {}).get(ticker, False),
        enriched_context=(enriched or {}).get(ticker),
    )


def _merge_thesis_results(
    existing: list[dict],
    updates: list[dict],
) -> list[dict]:
    by_ticker = {str(t.get("ticker", "")).upper(): t for t in existing}
    for thesis in updates:
        ticker = str(thesis.get("ticker", "")).upper()
        if ticker:
            by_ticker[ticker] = thesis
    order = [str(t.get("ticker", "")).upper() for t in existing]
    for thesis in updates:
        ticker = str(thesis.get("ticker", "")).upper()
        if ticker and ticker not in order:
            order.append(ticker)
    return [by_ticker[t] for t in order if t in by_ticker]


def _run_thesis_batch_with_ui(
    fundamentals_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    *,
    enriched: dict,
    relative_map: dict,
    comparative: dict | None = None,
    use_cache: bool,
    session_key: str,
    precomputed_signals: dict[str, dict] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    progress_slot = st.empty()
    status_slot = st.empty()

    def on_progress(current: int, total: int, ticker: str) -> None:
        progress_slot.progress(current / total if total else 0.0)
        status_slot.caption(f"Generating thesis for {ticker} ({current}/{total})...")

    with st.spinner("Generating investment theses and signals..."):
        theses = generate_theses_for_candidates(
            fundamentals_df,
            cache_get=db.get_cached_thesis if use_cache else None,
            cache_set=db.save_thesis_cache if use_cache else None,
            enriched_by_ticker=enriched or None,
            relative_context_by_ticker=relative_map,
            comparative_context_by_ticker=comparative or None,
            on_progress=on_progress,
        )
        if precomputed_signals:
            signals = precomputed_signals
        else:
            signals = _build_thesis_signals(fundamentals_df, theses, enriched, universe_df)

    progress_slot.empty()
    status_slot.empty()

    st.session_state[f"{session_key}_context"] = {
        "fundamentals_df": fundamentals_df,
        "universe_df": universe_df,
        "enriched": enriched,
        "relative_map": relative_map,
        "comparative": comparative,
        "use_cache": use_cache,
    }
    st.session_state[f"{session_key}_theses"] = theses
    st.session_state[f"{session_key}_signals"] = signals
    return theses, signals


def _render_thesis_section(
    fundamentals_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    theses: list[dict],
    signals: dict[str, dict],
    enriched: dict,
    *,
    session_key: str,
) -> None:
    short_names = _short_name_lookup(fundamentals_df)
    is_fund_flags = {
        str(row["ticker"]).upper(): is_fund(row)
        for _, row in fundamentals_df.iterrows()
    }

    failed = [
        str(t.get("ticker", "")).upper()
        for t in theses
        if _thesis_failed(t, enriched=enriched, is_fund_flags=is_fund_flags)
    ]
    ok_count = len(theses) - len(failed)
    if failed:
        st.info(
            f"Generated {ok_count}/{len(theses)} theses (failed: {', '.join(failed)})"
        )
    else:
        st.info(f"Generated {ok_count}/{len(theses)} theses.")

    retry_key = f"retry_failed_{session_key}"
    if failed and st.button(f"Retry failed theses ({len(failed)})", key=retry_key):
        ctx = st.session_state.get(f"{session_key}_context", {})
        retry_df = fundamentals_df[
            fundamentals_df["ticker"].str.upper().isin(failed)
        ]
        progress_slot = st.empty()
        status_slot = st.empty()

        def on_progress(current: int, total: int, ticker: str) -> None:
            progress_slot.progress(current / total if total else 0.0)
            status_slot.caption(f"Retrying thesis for {ticker} ({current}/{total})...")

        updates: list[dict] = []
        with st.spinner("Retrying failed theses..."):
            for idx, (_, row) in enumerate(retry_df.iterrows(), start=1):
                ticker = str(row["ticker"]).upper()
                on_progress(idx, len(retry_df), ticker)
                enrichment = (ctx.get("enriched") or enriched or {}).get(ticker)
                relative = (ctx.get("relative_map") or {}).get(ticker)
                comparative = (ctx.get("comparative") or {}).get(ticker)
                thesis = generate_thesis(
                    ticker,
                    row,
                    enrichment,
                    relative_context=relative,
                    comparative_context=comparative,
                )
                if not thesis.get("error") and ctx.get("use_cache"):
                    db.save_thesis_cache(ticker, thesis)
                updates.append(thesis)
            theses = _merge_thesis_results(theses, updates)
            signals = _build_thesis_signals(
                fundamentals_df,
                theses,
                ctx.get("enriched") or enriched,
                ctx.get("universe_df") or universe_df,
            )
            st.session_state[f"{session_key}_theses"] = theses
            st.session_state[f"{session_key}_signals"] = signals

        progress_slot.empty()
        status_slot.empty()
        st.rerun()

    st.subheader("Investment Theses")
    for thesis in theses:
        ticker_key = str(thesis.get("ticker", "")).upper()
        label = _thesis_expander_label(
            ticker_key,
            short_names.get(ticker_key),
            thesis,
            enriched=enriched,
            is_fund_flag=is_fund_flags.get(ticker_key, False),
        )
        with st.expander(label):
            _render_thesis_expander(
                thesis,
                signals.get(ticker_key),
                is_fund_flag=is_fund_flags.get(ticker_key, False),
            )


def _build_relative_context_map(
    fundamentals_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    tickers = fundamentals_df["ticker"].tolist()
    momentum = compute_momentum_scores(tickers)
    context_map: dict[str, dict] = {}
    for _, row in fundamentals_df.iterrows():
        ticker = str(row["ticker"]).upper()
        weight = (weights or {}).get(ticker) or (weights or {}).get(row["ticker"])
        context_map[ticker] = build_relative_context(
            ticker,
            row,
            universe_fundamentals=universe_df,
            momentum_score=momentum.get(ticker),
            portfolio_weight=weight,
        )
    return context_map


def _prepare_thesis_enrichment(
    fundamentals_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    alternates_df: pd.DataFrame | None = None,
) -> dict:
    from portfolio_advisor.fundamentals_finnhub import finnhub_configured

    if not finnhub_configured():
        return {}

    enrich_df = fundamentals_df.copy()
    if alternates_df is not None and not alternates_df.empty:
        extra = alternates_df[~alternates_df["ticker"].isin(enrich_df["ticker"])]
        if not extra.empty:
            enrich_df = pd.concat([enrich_df, extra], ignore_index=True)

    ticker_df = enrich_df[["ticker"]].drop_duplicates()
    return enrich_with_finnhub(ticker_df, fundamentals_df=enrich_df)


def _build_holdings_signals(
    fundamentals_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    enriched: dict,
) -> dict[str, dict]:
    sentiments = score_news_sentiments_for_tickers(
        enriched,
        cache_get=db.get_cached_news_sentiment,
        cache_set=db.save_news_sentiment_cache,
    )
    return build_signals_for_holdings(
        fundamentals_df=fundamentals_df,
        enriched_by_ticker=enriched,
        theses_by_ticker=None,
        news_sentiments=sentiments,
        universe_fundamentals=universe_df,
    )


def _build_thesis_signals(
    fundamentals_df: pd.DataFrame,
    theses: list[dict],
    enriched: dict,
    universe_df: pd.DataFrame,
) -> dict[str, dict]:
    theses_by_ticker = {
        str(t.get("ticker", "")).upper(): t for t in theses if not t.get("error")
    }
    sentiments = score_news_sentiments_for_tickers(
        enriched,
        cache_get=db.get_cached_news_sentiment,
        cache_set=db.save_news_sentiment_cache,
    )
    return build_signals_for_holdings(
        fundamentals_df=fundamentals_df,
        enriched_by_ticker=enriched,
        theses_by_ticker=theses_by_ticker,
        news_sentiments=sentiments,
        universe_fundamentals=universe_df,
    )


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
        if not _check_rate_limit("build"):
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

        max_weight_cap = RISK_PARAMS.get(risk_tolerance, RISK_PARAMS["moderate"])["max_weight"]
        st.info(
            f"Built with: **{risk_tolerance}** / **{style}** / **{horizon}** | "
            f"Max weight cap: **{max_weight_cap:.0%}**"
        )

        st.subheader("Screened Candidates")
        betas = candidates["beta"].dropna()
        if not betas.empty:
            st.caption(
                f"Beta range of picks: **{betas.min():.2f}** – **{betas.max():.2f}** "
                f"(risk filter: {risk_tolerance})"
            )
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
        render_allocation_pie(weights, title="Target Allocation")
        st.metric("Expected annual return", f"{stats['expected_return']:.1%}")
        st.metric("Volatility", f"{stats['volatility']:.1%}")
        st.metric("Sharpe ratio", f"{stats['sharpe_ratio']:.2f}")

        theses = []
        signals: dict[str, dict] = {}
        if generate_thesis:
            if not llm_available():
                st.warning("AI thesis generation skipped — no LLM API key configured.")
            else:
                comparative = find_sector_alternatives(
                    candidates,
                    universe_df,
                    risk_tolerance=risk_tolerance,
                    style=style,
                    exclude_sectors=exclude_sectors,
                )
                alt_rows = []
                for alt_list in comparative.values():
                    alt_rows.extend(alt_list)
                alt_df = pd.DataFrame(alt_rows) if alt_rows else pd.DataFrame()
                if not alt_df.empty and "ticker" in alt_df.columns:
                    alt_df = alt_df.drop_duplicates(subset=["ticker"])

                enriched = _prepare_thesis_enrichment(candidates, universe_df, alt_df)
                relative_map = _build_relative_context_map(
                    candidates, universe_df, weights=weights
                )
                theses, signals = _run_thesis_batch_with_ui(
                    candidates,
                    universe_df,
                    enriched=enriched,
                    relative_map=relative_map,
                    comparative=comparative or None,
                    use_cache=not enriched,
                    session_key="build",
                )
                _render_thesis_section(
                    candidates,
                    universe_df,
                    theses,
                    signals,
                    enriched,
                    session_key="build",
                )

        theses_for_save = []
        for thesis in theses:
            if thesis.get("error"):
                continue
            saved = dict(thesis)
            ticker_key = str(thesis.get("ticker", "")).upper()
            saved["conviction"] = (signals.get(ticker_key) or {}).get("conviction", "")
            theses_for_save.append(saved)

        portfolio_id = db.save_portfolio(
            capital=capital,
            risk_tolerance=risk_tolerance,
            style=style,
            horizon=horizon,
            excluded_sectors=exclude_sectors,
            weights=weights,
            theses=theses_for_save,
        )
        st.success(f"Portfolio saved (ID #{portfolio_id}).")

    _render_past_portfolios()


def _render_analyze_mode() -> None:
    from portfolio_advisor.webull_client import WebullConnectionError, webull_configured

    st.header("Analyze My Portfolio")

    holdings_source = st.radio(
        "Holdings source",
        ["Enter manually", "Connect Webull account"],
        horizontal=True,
    )
    use_webull = holdings_source == "Connect Webull account"

    if use_webull and not webull_configured():
        st.info(
            "Webull API credentials are not configured. Add `WEBULL_APP_KEY` and "
            "`WEBULL_APP_SECRET` to `.env` or Streamlit secrets. "
            "Apply at [developer.webull.com](https://developer.webull.com). "
            "Switch to manual entry to analyze without Webull."
        )

    default_holdings = pd.DataFrame(
        {"ticker": ["AAPL", "MSFT"], "shares": [10.0, 5.0]}
    )
    holdings_input = None
    if not use_webull:
        holdings_input = st.data_editor(
            default_holdings, num_rows="dynamic", use_container_width=True
        )

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
    rebalance_mode = "math_only"
    if analyze_action == "Get rebalance suggestions":
        rebalance_mode_label = st.radio(
            "Rebalance basis",
            list(REBALANCE_MODE_LABELS.values()),
            horizontal=True,
            key="analyze_rebalance_mode",
        )
        rebalance_mode = next(
            key
            for key, label in REBALANCE_MODE_LABELS.items()
            if label == rebalance_mode_label
        )
    analyze_clicked = st.button("Analyze Portfolio", type="primary")
    refresh_webull = False
    if use_webull and webull_configured():
        refresh_webull = st.button("Refresh from Webull")

    if not (analyze_clicked or refresh_webull):
        return

    if use_webull and not webull_configured():
        st.warning("Configure Webull credentials or switch to manual entry.")
        return

    if not _check_rate_limit("analyze"):
        _show_rate_limit_warning()
        return

    rows: list[dict] = []
    source = "webull" if use_webull else "manual"
    if source == "manual":
        assert holdings_input is not None
        rows = holdings_input.dropna(subset=["ticker"]).to_dict("records")
        rows = [r for r in rows if str(r.get("ticker", "")).strip()]
        if not rows:
            st.warning("Enter at least one ticker with shares.")
            return

    with st.spinner("Loading current positions..."):
        try:
            position_df, fundamentals_df, total_value, invalid_tickers = build_current_position_df(
                rows if source == "manual" else None,
                source=source,
            )
        except WebullConnectionError as exc:
            st.error(str(exc))
            return

    if invalid_tickers:
        st.warning(
            f"Couldn't find data for: {', '.join(invalid_tickers)} — check the ticker symbols."
        )

    if position_df.empty or total_value <= 0:
        st.warning("Could not compute portfolio value from the valid tickers. Check ticker symbols and shares.")
        return

    st.subheader("Current Positions")
    st.dataframe(position_df, use_container_width=True)
    if use_webull:
        st.caption(
            "Holdings imported from Webull. Prices are from yfinance (delayed/cached), not Webull quotes."
        )
    st.metric("Total portfolio value", f"${total_value:,.2f}")

    universe_df = cached_fetch_fundamentals(tuple(DEFAULT_UNIVERSE))
    rebalance_stats = None
    actions = pd.DataFrame()
    holdings_signals: dict[str, dict] = {}
    enriched_for_signals: dict = {}

    if analyze_action == "Get rebalance suggestions":
        if rebalance_mode == "signal_aligned":
            enriched_for_signals = _prepare_thesis_enrichment(fundamentals_df, universe_df)
            holdings_signals = _build_holdings_signals(
                fundamentals_df, universe_df, enriched_for_signals
            )
        with st.spinner("Computing rebalance suggestions..."):
            actions, rebalance_stats, opt_warning = compute_rebalance_actions(
                position_df,
                total_value,
                risk_tolerance,
                mode=rebalance_mode,
                signals=holdings_signals or None,
            )
        if opt_warning:
            st.warning(opt_warning)

    risk = classify_risk_level(
        position_df,
        fundamentals_df,
        volatility=rebalance_stats.get("volatility") if rebalance_stats else None,
    )
    st.subheader("Portfolio Risk Profile")
    render_risk_gauge(risk)
    st.write(risk["explanation"])
    st.caption(
        f"Weighted beta: {risk['weighted_beta']} | "
        f"Concentration index: {risk['concentration_index']}"
    )

    sectors = sector_concentration(position_df)
    render_sector_donut(sectors)

    if analyze_action == "Get rebalance suggestions" and not actions.empty:
        st.subheader("Rebalance Suggestions")
        st.caption(REBALANCE_MODE_DESCRIPTIONS[rebalance_mode])
        display_cols = [
            "ticker",
            "current_weight_pct",
            "target_weight_pct",
            "current_value",
            "target_value",
            "trade",
            "action",
        ]
        st.dataframe(actions[display_cols], use_container_width=True)
        render_rebalance_bar(actions)
        st.metric("Target expected return", f"{rebalance_stats['expected_return']:.1%}")
        st.metric("Target volatility", f"{rebalance_stats['volatility']:.1%}")
        st.metric("Target Sharpe ratio", f"{rebalance_stats['sharpe_ratio']:.2f}")

    if generate_thesis:
        if not llm_available():
            st.warning("AI thesis generation skipped — no LLM API key configured.")
        else:
            enriched = enriched_for_signals or _prepare_thesis_enrichment(
                fundamentals_df, universe_df
            )
            if not enriched:
                st.info(
                    "Finnhub not configured — theses use yfinance fundamentals only. "
                    "Add `FINNHUB_API_KEY` for analyst ratings, news, and earnings context."
                )
            weight_map = {
                str(row["ticker"]).upper(): float(row["weight"])
                for _, row in position_df.iterrows()
            }
            relative_map = _build_relative_context_map(
                fundamentals_df,
                universe_df,
                weights=weight_map,
            )
            theses, signals = _run_thesis_batch_with_ui(
                fundamentals_df,
                universe_df,
                enriched=enriched,
                relative_map=relative_map,
                comparative=None,
                use_cache=not enriched,
                session_key="analyze",
                precomputed_signals=holdings_signals or None,
            )
            _render_thesis_section(
                fundamentals_df,
                universe_df,
                theses,
                signals,
                enriched,
                session_key="analyze",
            )


def main() -> None:
    bootstrap_app()
    _init_session_state()

    if _dev_mode_enabled():
        print("DEV_MODE: rate limiting relaxed (100/hour per mode; set DEV_MODE_DISABLE_RATE_LIMIT=true to skip)")

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
