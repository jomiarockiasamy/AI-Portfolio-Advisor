"""Review current holdings and compute rebalance suggestions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from portfolio_advisor.data_fetcher import compute_daily_returns, fetch_price_history
from portfolio_advisor.optimizer import optimize_portfolio, portfolio_stats

REBALANCE_THRESHOLD_USD = 50.0


def _normalize_holdings(holdings: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Normalize tickers, drop invalid share counts, merge duplicates."""
    merged: Dict[str, float] = {}
    for item in holdings:
        ticker = str(item.get("ticker", "")).strip().upper()
        shares = item.get("shares", 0)
        try:
            shares = float(shares)
        except (TypeError, ValueError):
            shares = 0.0
        if ticker and shares > 0:
            merged[ticker] = merged.get(ticker, 0.0) + shares

    return [{"ticker": ticker, "shares": shares} for ticker, shares in merged.items()]


def _is_valid_price(price) -> bool:
    if price is None or pd.isna(price):
        return False
    try:
        return float(price) > 0
    except (TypeError, ValueError):
        return False


def build_current_position_df(
    holdings: List[Dict[str, float | str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, float, List[str]]:
    """Build position table with market values and weights."""
    from portfolio_advisor.data_fetcher import fetch_fundamentals

    empty = pd.DataFrame(columns=["ticker", "shares", "price", "market_value", "weight", "sector"])
    cleaned = _normalize_holdings(holdings)

    if not cleaned:
        return empty, pd.DataFrame(), 0.0, []

    tickers = [item["ticker"] for item in cleaned]
    fundamentals = fetch_fundamentals(tickers)

    if fundamentals.empty or "ticker" not in fundamentals.columns:
        return empty, pd.DataFrame(), 0.0, tickers

    fundamentals = fundamentals.set_index("ticker", drop=False)
    invalid_tickers: List[str] = []
    valid_items: List[Dict[str, float | str]] = []

    for item in cleaned:
        ticker = item["ticker"]
        if ticker not in fundamentals.index:
            invalid_tickers.append(ticker)
            continue
        row = fundamentals.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not _is_valid_price(row.get("currentPrice")):
            invalid_tickers.append(ticker)
            continue
        valid_items.append(item)

    if not valid_items:
        return empty, pd.DataFrame(), 0.0, invalid_tickers

    rows = []
    for item in valid_items:
        ticker = item["ticker"]
        row = fundamentals.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        price = float(row["currentPrice"])
        shares = float(item["shares"])
        market_value = price * shares
        rows.append(
            {
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "market_value": market_value,
                "sector": row.get("sector") or "Unknown",
            }
        )

    position_df = pd.DataFrame(rows)
    total_value = float(position_df["market_value"].sum())
    if total_value > 0:
        position_df["weight"] = position_df["market_value"] / total_value
    else:
        position_df["weight"] = 0.0

    valid_tickers = position_df["ticker"].tolist()
    valid_fundamentals = fundamentals[fundamentals["ticker"].isin(valid_tickers)].reset_index(drop=True)

    return position_df, valid_fundamentals, total_value, invalid_tickers


def sector_concentration(position_df: pd.DataFrame) -> pd.Series:
    """Return sector weights sorted descending."""
    if position_df.empty:
        return pd.Series(dtype=float)
    return (
        position_df.groupby("sector")["weight"]
        .sum()
        .sort_values(ascending=False)
    )


def compute_rebalance_actions(
    position_df: pd.DataFrame,
    total_value: float,
    risk_tolerance: str,
    lookback_period: str = "2y",
) -> Tuple[pd.DataFrame, Dict[str, float], str | None]:
    """Compare current weights to optimized targets and suggest actions."""
    empty = pd.DataFrame(
        columns=[
            "ticker",
            "current_weight",
            "target_weight",
            "current_value",
            "target_value",
            "delta_value",
            "action",
        ]
    )
    empty_stats = {"expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

    if position_df.empty or total_value <= 0:
        return empty, empty_stats, None

    tickers = position_df["ticker"].tolist()
    prices = fetch_price_history(tickers, period=lookback_period)
    returns = compute_daily_returns(prices)
    target_weights, opt_warning = optimize_portfolio(returns, risk_tolerance)

    rows = []
    for _, row in position_df.iterrows():
        ticker = row["ticker"]
        current_weight = float(row["weight"])
        current_value = float(row["market_value"])
        target_weight = float(target_weights.get(ticker, 0.0))
        target_value = target_weight * total_value
        delta_value = target_value - current_value

        if abs(delta_value) < REBALANCE_THRESHOLD_USD:
            action = "HOLD"
        elif delta_value > 0:
            action = "BUY"
        else:
            action = "SELL"

        rows.append(
            {
                "ticker": ticker,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "current_value": current_value,
                "target_value": target_value,
                "delta_value": delta_value,
                "action": action,
            }
        )

    action_df = pd.DataFrame(rows)
    stats = portfolio_stats(target_weights, returns)
    return action_df, stats, opt_warning


if __name__ == "__main__":
    import numpy as np
    from unittest.mock import patch

    mock_fundamentals = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "currentPrice": [150.0, 300.0],
            "sector": ["Technology", "Technology"],
        }
    )
    mock_prices = pd.DataFrame(
        {
            "AAPL": np.linspace(140, 160, 60),
            "MSFT": np.linspace(280, 320, 60),
        }
    )

    holdings = [{"ticker": "AAPL", "shares": 10}, {"ticker": "MSFT", "shares": 5}]

    with patch("portfolio_advisor.data_fetcher.fetch_fundamentals", return_value=mock_fundamentals), patch(
        "portfolio_advisor.data_fetcher.fetch_price_history", return_value=mock_prices
    ):
        position_df, _, total_value, invalid = build_current_position_df(holdings)
        assert invalid == []
        assert abs(position_df["weight"].sum() - 1.0) < 1e-6

        sectors = sector_concentration(position_df)
        assert abs(sectors.sum() - 1.0) < 1e-6

        actions, stats, _ = compute_rebalance_actions(position_df, total_value, "moderate")
        assert set(actions["action"]).issubset({"BUY", "SELL", "HOLD"})
        assert "sharpe_ratio" in stats

        mixed = [{"ticker": "AAPL", "shares": 10}, {"ticker": "ZZZZQQQ", "shares": 5}]
        pos2, _, val2, invalid2 = build_current_position_df(mixed)
        assert invalid2 == ["ZZZZQQQ"]
        assert len(pos2) == 1
        assert pos2.iloc[0]["ticker"] == "AAPL"
        assert val2 > 0

        dupes = [{"ticker": "aapl", "shares": 5}, {"ticker": "AAPL", "shares": 5}]
        pos3, _, _, _ = build_current_position_df(dupes)
        assert len(pos3) == 1
        assert pos3.iloc[0]["shares"] == 10

    print("holdings checkpoint passed")
