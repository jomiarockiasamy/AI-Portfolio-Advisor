"""Portfolio-level risk metrics and plain-English classification."""

from __future__ import annotations

from typing import Any

import pandas as pd

from portfolio_advisor.data_fetcher import compute_daily_returns, fetch_price_history
from portfolio_advisor.holdings import sector_concentration


def weighted_beta(position_df: pd.DataFrame, fundamentals_df: pd.DataFrame | None = None) -> float:
    if position_df.empty:
        return 0.0

    beta_map: dict[str, float] = {}
    if fundamentals_df is not None and not fundamentals_df.empty:
        for _, row in fundamentals_df.iterrows():
            beta = row.get("beta")
            if beta is not None and not pd.isna(beta):
                beta_map[str(row["ticker"]).upper()] = float(beta)

    total = 0.0
    for _, row in position_df.iterrows():
        ticker = str(row["ticker"]).upper()
        weight = float(row["weight"])
        beta = beta_map.get(ticker, 1.0)
        total += weight * beta
    return round(total, 2)


def concentration_index(position_df: pd.DataFrame) -> float:
    """Herfindahl-Hirschman Index: sum of squared weights (higher = more concentrated)."""
    if position_df.empty:
        return 0.0
    weights = position_df["weight"].astype(float)
    return round(float((weights ** 2).sum()), 4)


def portfolio_volatility(position_df: pd.DataFrame, period: str = "2y") -> float | None:
    if position_df.empty:
        return None
    tickers = position_df["ticker"].tolist()
    prices = fetch_price_history(tickers, period=period)
    if prices.empty:
        return None
    returns = compute_daily_returns(prices)
    if returns.empty:
        return None

    weights = position_df.set_index("ticker")["weight"].astype(float)
    aligned = returns[[t for t in weights.index if t in returns.columns]]
    if aligned.empty:
        return None
    w = weights.reindex(aligned.columns).fillna(0.0)
    if w.sum() > 0:
        w = w / w.sum()
    port_returns = aligned.mul(w, axis=1).sum(axis=1)
    return round(float(port_returns.std() * (252 ** 0.5)), 4)


def classify_risk_level(
    position_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None = None,
    volatility: float | None = None,
) -> dict[str, Any]:
    beta = weighted_beta(position_df, fundamentals_df)
    hhi = concentration_index(position_df)
    vol = volatility if volatility is not None else portfolio_volatility(position_df)

    sectors = sector_concentration(position_df)
    top_sector = sectors.index[0] if not sectors.empty else "Unknown"
    top_sector_weight = float(sectors.iloc[0]) if not sectors.empty else 0.0

    score = 0
    if beta >= 1.3:
        score += 2
    elif beta >= 1.0:
        score += 1
    if hhi >= 0.25:
        score += 2
    elif hhi >= 0.15:
        score += 1
    if vol is not None and vol >= 0.25:
        score += 1
    elif vol is not None and vol >= 0.18:
        score += 0.5

    if score >= 3:
        label = "Aggressive"
    elif score >= 1.5:
        label = "Moderate"
    else:
        label = "Conservative"

    vol_text = f"{vol:.1%} annualized volatility" if vol is not None else "volatility unavailable"
    explanation = (
        f"Your portfolio's weighted beta of {beta:.2f}, concentration index of {hhi:.2f}, "
        f"and {top_sector_weight:.0%} in {top_sector} ({vol_text}) "
        f"place this portfolio in {label} territory."
    )

    gauge_value = min(100.0, max(0.0, beta * 30 + hhi * 120 + (vol or 0.15) * 100))

    return {
        "label": label,
        "weighted_beta": beta,
        "concentration_index": hhi,
        "volatility": vol,
        "top_sector": top_sector,
        "top_sector_weight": top_sector_weight,
        "explanation": explanation,
        "gauge_value": round(gauge_value, 1),
    }


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "ticker": ["NVDA", "AMD", "MSFT", "TSLA"],
            "weight": [0.35, 0.25, 0.25, 0.15],
            "market_value": [3500, 2500, 2500, 1500],
            "sector": ["Technology", "Technology", "Technology", "Consumer Cyclical"],
        }
    )
    fund = pd.DataFrame(
        {
            "ticker": ["NVDA", "AMD", "MSFT", "TSLA"],
            "beta": [2.2, 1.9, 1.1, 2.0],
        }
    )
    result = classify_risk_level(sample, fund, volatility=0.30)
    assert result["label"] == "Aggressive", result
    print(result)
    print("risk_profile checkpoint passed")
