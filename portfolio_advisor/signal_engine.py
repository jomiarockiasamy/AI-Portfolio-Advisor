"""Composite BUY/HOLD/SELL signal engine with explainable sub-scores."""

from __future__ import annotations

from typing import Any

import pandas as pd

from portfolio_advisor.data_fetcher import (
    DEFAULT_UNIVERSE,
    compute_momentum_scores,
    fetch_fundamentals,
    normalize_ticker,
)
from portfolio_advisor.formatters import sanitize_peg

WEIGHTS: dict[str, float] = {
    "valuation": 0.22,
    "growth": 0.22,
    "analyst": 0.22,
    "news_sentiment": 0.17,
    "momentum": 0.17,
}
BUY_THRESHOLD = 65.0
SELL_THRESHOLD = 40.0
NEWS_NEUTRAL = 50.0
ANALYST_NEUTRAL = 50.0

COMPONENT_LABELS: dict[str, str] = {
    "valuation": "Valuation (vs sector P/E, PEG)",
    "growth": "Growth (revenue & earnings)",
    "analyst": "Analyst ratings",
    "news_sentiment": "News sentiment",
    "momentum": "Price momentum",
}

COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "valuation": "Compares trailing P/E and PEG to the sector median; lower multiples vs peers score higher.",
    "growth": "Blends revenue and earnings growth; stronger growth scores higher.",
    "analyst": "Finnhub analyst rating balance (buys vs sells); more bullish consensus scores higher.",
    "news_sentiment": "LLM-scored filtered headlines; defaults to neutral when news is insufficient.",
    "momentum": "50/200-day moving-average trend; defaults to neutral when price history is thin.",
}

FUND_SCORING_NOTE = (
    "ETFs and funds use the same composite engine today. Growth and analyst sub-scores may be thin; "
    "expense ratio, AUM, beta, and index exposure matter more for fund theses (fund_role, not moat). "
    "Fund-specific scoring weights are planned for a future release."
)


def compute_sector_medians(universe_fundamentals: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    """Compute sector P/E and PEG medians from DEFAULT_UNIVERSE, not held tickers."""
    df = universe_fundamentals
    if df is None or df.empty:
        df = fetch_fundamentals(DEFAULT_UNIVERSE)
    if df.empty or "sector" not in df.columns:
        return {}

    medians: dict[str, dict[str, float]] = {}
    for sector, group in df.groupby("sector"):
        pe = group["trailingPE"].dropna()
        peg = group["pegRatio"].dropna()
        medians[str(sector)] = {
            "trailingPE": float(pe.median()) if len(pe) else float("nan"),
            "pegRatio": float(peg.median()) if len(peg) else float("nan"),
        }
    return medians


def score_valuation(
    row: pd.Series | dict,
    sector_medians: dict[str, dict[str, float]],
) -> float:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    sector = str(row.get("sector") or "Unknown")
    med = sector_medians.get(sector, {})
    pe = row.get("trailingPE")
    peg = row.get("pegRatio")
    pe_med = med.get("trailingPE")
    peg_med = med.get("pegRatio")

    scores: list[float] = []
    if pe and pe_med and pe > 0 and pe_med > 0:
        ratio = pe_med / pe
        scores.append(max(0.0, min(100.0, 50.0 + (ratio - 1.0) * 50.0)))
    peg_val, peg_flag = sanitize_peg(peg, row.get("earningsGrowth"))
    if peg_val and peg_med and peg_val > 0 and peg_med > 0 and not peg_flag:
        ratio = peg_med / peg_val
        scores.append(max(0.0, min(100.0, 50.0 + (ratio - 1.0) * 50.0)))

    return float(sum(scores) / len(scores)) if scores else 50.0


def score_growth(row: pd.Series | dict) -> float:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    rev = row.get("revenueGrowth")
    earn = row.get("earningsGrowth")
    parts: list[float] = []
    for val in (rev, earn):
        if val is None or pd.isna(val):
            continue
        parts.append(max(0.0, min(100.0, 50.0 + float(val) * 100.0)))
    return float(sum(parts) / len(parts)) if parts else 50.0


def score_analyst(ratings: dict[str, Any] | None) -> float:
    if not ratings:
        return ANALYST_NEUTRAL
    bullish = (
        float(ratings.get("strong_buy", 0) or 0)
        + float(ratings.get("buy", 0) or 0)
    )
    bearish = (
        float(ratings.get("strong_sell", 0) or 0)
        + float(ratings.get("sell", 0) or 0)
    )
    total = bullish + bearish + float(ratings.get("hold", 0) or 0)
    if total <= 0:
        return ANALYST_NEUTRAL
    net = (bullish - bearish) / total
    return max(0.0, min(100.0, 50.0 + net * 50.0))


def score_news_sentiment_component(sentiment: dict[str, Any] | None) -> float:
    if not sentiment:
        return NEWS_NEUTRAL
    if sentiment.get("insufficient_data"):
        return NEWS_NEUTRAL
    raw = float(sentiment.get("score", 0.0))
    return max(0.0, min(100.0, 50.0 + raw * 50.0))


def conviction_from_composite(score: float) -> str:
    if score >= BUY_THRESHOLD:
        return "high"
    if score <= SELL_THRESHOLD:
        return "low"
    return "medium"


def _signal_label(composite: float) -> str:
    if composite >= BUY_THRESHOLD:
        return "BUY"
    if composite <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def compute_signal(
    ticker: str,
    fundamentals_row: pd.Series | dict,
    analyst_ratings: dict[str, Any] | None = None,
    news_sentiment: dict[str, Any] | None = None,
    momentum_score: float | None = None,
    sector_medians: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Return composite signal with per-component breakdown."""
    medians = sector_medians or compute_sector_medians()
    components_raw = {
        "valuation": score_valuation(fundamentals_row, medians),
        "growth": score_growth(fundamentals_row),
        "analyst": score_analyst(analyst_ratings),
        "news_sentiment": score_news_sentiment_component(news_sentiment),
        "momentum": float(momentum_score if momentum_score is not None else 50.0),
    }

    weighted: dict[str, float] = {}
    composite = 0.0
    for name, weight in WEIGHTS.items():
        contribution = components_raw[name] * weight
        weighted[name] = contribution
        composite += contribution

    composite = round(composite, 1)
    breakdown = []
    for name in WEIGHTS:
        breakdown.append(
            {
                "component": name,
                "score": round(components_raw[name], 1),
                "weight": WEIGHTS[name],
                "contribution": round(weighted[name], 1),
            }
        )

    return {
        "ticker": normalize_ticker(ticker),
        "composite_score": composite,
        "signal": _signal_label(composite),
        "conviction": conviction_from_composite(composite),
        "components": breakdown,
        "news_insufficient_data": bool((news_sentiment or {}).get("insufficient_data")),
    }


def build_signals_for_holdings(
    fundamentals_df: pd.DataFrame,
    enriched_by_ticker: dict[str, dict] | None = None,
    theses_by_ticker: dict[str, dict] | None = None,
    news_sentiments: dict[str, dict] | None = None,
    universe_fundamentals: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Build signals for each row in fundamentals_df."""
    enriched = enriched_by_ticker or {}
    sentiments = news_sentiments or {}
    sector_medians = compute_sector_medians(universe_fundamentals)

    tickers = [normalize_ticker(t) for t in fundamentals_df["ticker"].tolist()]
    momentum = compute_momentum_scores(tickers)

    signals: dict[str, dict] = {}
    for _, row in fundamentals_df.iterrows():
        ticker = normalize_ticker(str(row["ticker"]))
        enrichment = enriched.get(ticker, {})
        signals[ticker] = compute_signal(
            ticker=ticker,
            fundamentals_row=row,
            analyst_ratings=enrichment.get("analyst_ratings"),
            news_sentiment=sentiments.get(ticker),
            momentum_score=momentum.get(ticker, 50.0),
            sector_medians=sector_medians,
        )
    return signals


if __name__ == "__main__":
    universe = fetch_fundamentals(DEFAULT_UNIVERSE)
    sample = universe[universe["ticker"].isin(["NVDA", "JNJ", "GPRO"])].copy()
    if len(sample) < 3:
        sample = universe.head(3)

    medians = compute_sector_medians(universe)
    assert medians, "sector medians should not be empty"

    scores = []
    for _, row in sample.iterrows():
        sig = compute_signal(row["ticker"], row, sector_medians=medians)
        scores.append(sig["composite_score"])
        assert sig["conviction"] == conviction_from_composite(sig["composite_score"])
        print(row["ticker"], sig["signal"], sig["composite_score"], sig["conviction"])

    assert len(set(scores)) > 1 or len(sample) == 1, "Expected differentiated scores"
    print("signal_engine checkpoint passed")
