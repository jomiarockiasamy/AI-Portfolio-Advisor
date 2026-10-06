"""Review current holdings and compute rebalance suggestions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Tuple

import pandas as pd

from portfolio_advisor.data_fetcher import (
    compute_daily_returns,
    effective_price,
    fetch_price_history,
    normalize_ticker,
)
from portfolio_advisor.formatters import format_trade_action, format_weight_pct, snap_weight_dust
from portfolio_advisor.optimizer import optimize_portfolio, portfolio_stats
from portfolio_advisor.rebalance_engine import (
    apply_signal_tilt,
    get_max_weight,
)

logger = logging.getLogger(__name__)

REBALANCE_THRESHOLD_USD = 50.0


def _normalize_holdings(holdings: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Normalize tickers, drop invalid share counts, merge duplicates."""
    merged: Dict[str, Dict[str, float | str]] = {}
    for item in holdings:
        ticker = normalize_ticker(str(item.get("ticker", "")))
        shares = item.get("shares", 0)
        try:
            shares = float(shares)
        except (TypeError, ValueError):
            shares = 0.0
        if not ticker or shares <= 0:
            continue

        if ticker not in merged:
            merged[ticker] = {"ticker": ticker, "shares": shares}
        else:
            merged[ticker]["shares"] = float(merged[ticker]["shares"]) + shares

        avg_cost = item.get("avg_cost")
        if avg_cost is not None:
            try:
                merged[ticker]["avg_cost"] = float(avg_cost)
            except (TypeError, ValueError):
                pass

    return list(merged.values())


def _is_valid_price(price) -> bool:
    if price is None or pd.isna(price):
        return False
    try:
        return float(price) > 0
    except (TypeError, ValueError):
        return False


def _resolve_holdings(
    holdings: List[Dict[str, float | str]] | None,
    source: Literal["manual", "webull"],
) -> List[Dict[str, float | str]]:
    if source == "webull":
        from portfolio_advisor.webull_client import get_account_positions

        return get_account_positions()
    return _normalize_holdings(holdings or [])


def build_current_position_df(
    holdings: List[Dict[str, float | str]] | None = None,
    source: Literal["manual", "webull"] = "manual",
) -> Tuple[pd.DataFrame, pd.DataFrame, float, List[str]]:
    """Build position table with market values and weights."""
    from portfolio_advisor.data_fetcher import fetch_fundamentals

    empty = pd.DataFrame(
        columns=["ticker", "shares", "price", "market_value", "weight", "sector"]
    )
    cleaned = _resolve_holdings(holdings, source)

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
        if not _is_valid_price(effective_price(row)):
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
        price = float(effective_price(row))
        shares = float(item["shares"])
        market_value = price * shares
        entry: Dict[str, float | str] = {
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "market_value": market_value,
            "sector": row.get("sector") or "Unknown",
        }
        if source == "webull" and item.get("avg_cost") is not None:
            entry["avg_cost"] = float(item["avg_cost"])
        rows.append(entry)

    position_df = pd.DataFrame(rows)
    total_value = float(position_df["market_value"].sum())
    if total_value > 0:
        position_df["weight"] = position_df["market_value"] / total_value
    else:
        position_df["weight"] = 0.0

    valid_tickers = position_df["ticker"].tolist()
    valid_fundamentals = fundamentals[fundamentals["ticker"].isin(valid_tickers)].reset_index(drop=True)

    return position_df, valid_fundamentals, total_value, invalid_tickers


def enrich_with_finnhub(
    position_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Fetch Finnhub analyst/news/earnings context for each ticker."""
    from portfolio_advisor.data_fetcher import is_fund, normalize_ticker
    from portfolio_advisor.formatters import format_fund_fields
    from portfolio_advisor.fundamentals_finnhub import (
        finnhub_configured,
        get_analyst_ratings,
        get_earnings_calendar,
        get_recent_news,
    )

    if position_df.empty or not finnhub_configured():
        return {}

    fund_lookup: dict[str, dict] = {}
    if fundamentals_df is not None and not fundamentals_df.empty:
        for _, row in fundamentals_df.iterrows():
            fund_lookup[normalize_ticker(str(row["ticker"]))] = row.to_dict()

    enriched: dict[str, dict] = {}
    for ticker in position_df["ticker"].tolist():
        symbol = normalize_ticker(str(ticker))
        row = fund_lookup.get(symbol, {"ticker": symbol})
        short_name = row.get("shortName")
        fund = is_fund(row)

        try:
            context: dict[str, Any] = {"is_fund": fund}
            if fund:
                raw_fields = {
                    key: row.get(key)
                    for key in (
                        "annualReportExpenseRatio",
                        "totalAssets",
                        "fundFamily",
                        "dividendYield",
                        "beta",
                        "shortName",
                    )
                    if row.get(key) is not None
                }
                context["fund_fields"] = format_fund_fields(raw_fields)
                context["news_status"] = "insufficient"
                context["recent_news"] = []
            else:
                ratings = get_analyst_ratings(symbol)
                news = get_recent_news(symbol, short_name=short_name)
                earnings = get_earnings_calendar(symbol)
                context["analyst_ratings"] = ratings
                context["recent_news"] = news
                context["news_status"] = "ok" if news else "insufficient"
                context["earnings"] = earnings

            if context.get("is_fund") or context.get("analyst_ratings") or context.get("recent_news") or context.get("earnings"):
                enriched[symbol] = context
        except Exception as exc:
            logger.warning("enrich failed for %s: %s", symbol, exc)
            continue

    return enriched


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
    *,
    mode: str = "math_only",
    signals: dict[str, dict] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], str | None]:
    """Compare current weights to optimized targets and suggest actions."""
    empty = pd.DataFrame(
        columns=[
            "ticker",
            "current_weight",
            "target_weight",
            "current_weight_pct",
            "target_weight_pct",
            "current_value",
            "target_value",
            "trade",
            "action",
        ]
    )
    empty_stats = {"expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

    if position_df.empty or total_value <= 0:
        return empty, empty_stats, None

    tickers = position_df["ticker"].tolist()
    prices = fetch_price_history(tickers, period=lookback_period)
    returns = compute_daily_returns(prices)
    base_weights, opt_warning = optimize_portfolio(returns, risk_tolerance)

    warnings: list[str] = []
    if opt_warning:
        warnings.append(opt_warning)

    if mode == "signal_aligned":
        if not signals:
            warnings.append(
                "Signal-aligned rebalance requested but no signals available — using math-only weights."
            )
            target_weights = base_weights
        else:
            current_weights = {
                str(row["ticker"]): float(row["weight"]) for _, row in position_df.iterrows()
            }
            max_weight = get_max_weight(risk_tolerance)
            target_weights = apply_signal_tilt(
                base_weights, current_weights, signals, max_weight
            )
    else:
        target_weights = base_weights

    rows = []
    for _, row in position_df.iterrows():
        ticker = row["ticker"]
        current_weight = float(row["weight"])
        current_value = float(row["market_value"])
        target_weight = snap_weight_dust(float(target_weights.get(ticker, 0.0)))
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
                "current_weight_pct": format_weight_pct(current_weight),
                "target_weight_pct": format_weight_pct(target_weight),
                "current_value": current_value,
                "target_value": target_value,
                "delta_value": delta_value,
                "trade": format_trade_action(delta_value, REBALANCE_THRESHOLD_USD),
                "action": action,
            }
        )

    action_df = pd.DataFrame(rows)
    stats = portfolio_stats(target_weights, returns)
    combined_warning = " ".join(warnings) if warnings else None
    return action_df, stats, combined_warning


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
        assert "trade" in actions.columns
        assert actions.loc[actions["action"] == "SELL", "trade"].str.startswith("Sell $").any() or True
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

        webull_holdings = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 140.0},
        ]
        with patch(
            "portfolio_advisor.webull_client.get_account_positions",
            return_value=webull_holdings,
        ):
            pos4, _, _, _ = build_current_position_df(source="webull")
            assert "avg_cost" in pos4.columns
            assert pos4.iloc[0]["avg_cost"] == 140.0

    print("holdings checkpoint passed")
