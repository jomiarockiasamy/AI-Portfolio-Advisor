"""Webull Trading API client — positions import only (no market-data quotes)."""

from __future__ import annotations

import os
from typing import Any

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient


class WebullConnectionError(Exception):
    """Raised when Webull Trading API calls fail or credentials are invalid."""


def webull_configured() -> bool:
    """Return True when both Webull app key and secret are set."""
    return bool(os.environ.get("WEBULL_APP_KEY") and os.environ.get("WEBULL_APP_SECRET"))


def _webull_host() -> str:
    env = os.environ.get("WEBULL_ENV", "sandbox").strip().lower()
    if env in {"production", "prod", "live"}:
        return "api.webull.com"
    return "api.sandbox.webull.com"


def _build_api_client() -> ApiClient:
    app_key = os.environ.get("WEBULL_APP_KEY", "").strip()
    app_secret = os.environ.get("WEBULL_APP_SECRET", "").strip()
    if not app_key or not app_secret:
        raise WebullConnectionError(
            "Webull credentials not configured. Set WEBULL_APP_KEY and WEBULL_APP_SECRET."
        )
    api_client = ApiClient(app_key, app_secret, "us")
    api_client.add_endpoint("us", _webull_host())
    return api_client


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _extract_ticker(position: dict[str, Any]) -> str | None:
    from portfolio_advisor.data_fetcher import normalize_ticker

    symbol = position.get("symbol")
    if symbol:
        return normalize_ticker(str(symbol))

    ticker_obj = position.get("ticker")
    if isinstance(ticker_obj, dict):
        for key in ("symbol", "disSymbol"):
            val = ticker_obj.get(key)
            if val:
                return normalize_ticker(str(val))
    return None


def _extract_shares(position: dict[str, Any]) -> float | None:
    for key in ("qty", "quantity", "position", "shares"):
        if key in position:
            shares = _parse_float(position.get(key))
            if shares is not None:
                return shares
    return None


def _extract_avg_cost(position: dict[str, Any]) -> float | None:
    for key in ("avg_cost", "costPrice", "average_cost", "cost_price", "cost"):
        if key in position:
            cost = _parse_float(position.get(key))
            if cost is not None:
                return cost
    return None


def _normalize_position(position: dict[str, Any]) -> dict[str, float | str] | None:
    ticker = _extract_ticker(position)
    shares = _extract_shares(position)
    if not ticker or shares is None:
        return None

    result: dict[str, float | str] = {"ticker": ticker, "shares": shares}
    avg_cost = _extract_avg_cost(position)
    if avg_cost is not None:
        result["avg_cost"] = avg_cost
    return result


def _positions_from_response(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("positions", "holdings", "data"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]

    return []


def _resolve_account_id(trade_client: TradeClient) -> str:
    configured = os.environ.get("WEBULL_ACCOUNT_ID", "").strip()
    if configured:
        return configured

    response = trade_client.account_v2.get_account_list()
    if response.status_code != 200:
        raise WebullConnectionError(
            f"Webull account list failed ({response.status_code}): {response.text}"
        )

    accounts = response.json()
    if not isinstance(accounts, list) or not accounts:
        raise WebullConnectionError("Webull returned no accounts for these credentials.")

    first = accounts[0]
    if not isinstance(first, dict):
        raise WebullConnectionError("Unexpected Webull account list format.")

    account_id = first.get("account_id") or first.get("accountId")
    if not account_id:
        raise WebullConnectionError("Could not determine Webull account_id from account list.")

    return str(account_id)


def get_account_positions() -> list[dict[str, float | str]]:
    """Fetch current stock positions from Webull Trading API.

    Returns a list of dicts: ``ticker``, ``shares``, optional ``avg_cost``.
    Pricing is intentionally not fetched here — use yfinance in holdings.py.
    """
    try:
        api_client = _build_api_client()
        trade_client = TradeClient(api_client)
        account_id = _resolve_account_id(trade_client)

        response = trade_client.account_v2.get_account_position(account_id)
        if response.status_code != 200:
            raise WebullConnectionError(
                f"Webull positions request failed ({response.status_code}): {response.text}"
            )

        raw_positions = _positions_from_response(response.json())
        merged: dict[str, dict[str, float | str]] = {}

        for raw in raw_positions:
            normalized = _normalize_position(raw)
            if not normalized:
                continue
            ticker = str(normalized["ticker"])
            shares = float(normalized["shares"])
            if ticker in merged:
                merged[ticker]["shares"] = float(merged[ticker]["shares"]) + shares
            else:
                merged[ticker] = normalized

        return list(merged.values())
    except WebullConnectionError:
        raise
    except Exception as exc:
        raise WebullConnectionError(f"Webull connection failed: {exc}") from exc


if __name__ == "__main__":
    quote_attrs = [name for name in dir(__import__(__name__)) if "quote" in name.lower()]
    assert quote_attrs == [], f"Quote functions must not exist: {quote_attrs}"

    print("webull_client: positions-only module (no DataClient / quote helpers).")
    print("Pricing decision: see portfolio_advisor/WEBULL_API_NOTES.md")

    if webull_configured():
        positions = get_account_positions()
        print(f"Fetched {len(positions)} position(s) from Webull.")
        for pos in positions[:5]:
            print(pos)
    else:
        print("WEBULL_APP_KEY / WEBULL_APP_SECRET not set — skipping live positions checkpoint.")
