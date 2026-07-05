"""Fetch price and fundamental data via yfinance."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, TypeVar

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_UNIVERSE: list[str] = [
    # Technology
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "META",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "LLY",
    # Financials
    "JPM",
    "BAC",
    "V",
    "MA",
    # Consumer
    "AMZN",
    "WMT",
    "PG",
    "KO",
    "PEP",
    # Energy / Industrials
    "XOM",
    "CVX",
    "CAT",
    "HON",
    # Communications
    "DIS",
    "NFLX",
    "T",
]

FUNDAMENTAL_FIELDS: list[str] = [
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToBook",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "dividendYield",
    "marketCap",
    "beta",
    "currentPrice",
    "sector",
    "shortName",
]


def _retry_once(fn: Callable[..., T], *args, **kwargs) -> T:
    """Try once, retry after a short sleep on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as first_exc:
        logger.warning("Retrying after error: %s", first_exc)
        time.sleep(1)
        return fn(*args, **kwargs)


def _fetch_ticker_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info or {}


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """Pull fundamental metrics for each ticker; skip failures individually."""
    rows: dict[str, dict] = {}

    for ticker in tickers:
        try:
            info = _retry_once(_fetch_ticker_info, ticker)
            row = {field: info.get(field) for field in FUNDAMENTAL_FIELDS}
            row["ticker"] = ticker
            rows[ticker] = row
        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker, exc)

    if not rows:
        return pd.DataFrame(columns=["ticker", *FUNDAMENTAL_FIELDS])

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reset_index(drop=True)


def _download_prices(ticker_list: list[str], period: str) -> pd.DataFrame:
    return yf.download(
        ticker_list,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )


def fetch_price_history(
    tickers: Iterable[str],
    period: str = "2y",
) -> pd.DataFrame:
    """Return wide DataFrame of adjusted close prices."""
    ticker_list = list(tickers)
    if not ticker_list:
        return pd.DataFrame()

    try:
        data = _retry_once(_download_prices, ticker_list, period)
    except Exception as exc:
        logger.warning("Failed to fetch price history: %s", exc)
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if len(ticker_list) == 1:
        close = data["Close"].to_frame(name=ticker_list[0])
        return close.dropna(how="all")

    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=ticker_list[0])
    return close.dropna(how="all")


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from a price DataFrame."""
    return price_df.pct_change().dropna()


if __name__ == "__main__":
    df = fetch_fundamentals(["AAPL", "MSFT"])
    print(df[["ticker", "shortName", "sector", "trailingPE", "beta"]])
    assert len(df) == 2, "Expected 2 rows"
    assert "trailingPE" in df.columns, "Missing trailingPE column"

    invalid = fetch_fundamentals(["INVALID1", "INVALID2"])
    assert isinstance(invalid, pd.DataFrame)
    print("data_fetcher checkpoint passed")
