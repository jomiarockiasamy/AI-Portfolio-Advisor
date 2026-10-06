"""Fetch price and fundamental data via yfinance."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, TypeVar

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

T = TypeVar("T")

TICKER_ALIASES: dict[str, str] = {
    "BRKB": "BRK-B",
    "BFB": "BF-B",
}

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
    "quoteType",
    "annualReportExpenseRatio",
    "totalAssets",
    "fundFamily",
]

KNOWN_ETF_TICKERS: set[str] = {
    "QQQ",
    "QQQM",
    "VOO",
    "IVV",
    "SPY",
    "VTI",
    "VBR",
    "VYM",
    "IWD",
    "IWF",
    "VEA",
    "VWO",
    "BND",
    "AGG",
}

MOMENTUM_MIN_HISTORY_DAYS = 200
MOMENTUM_NEUTRAL_SCORE = 50.0


def normalize_ticker(t: str) -> str:
    """Map broker-style symbols to Yahoo Finance tickers."""
    cleaned = str(t).strip().upper()
    return TICKER_ALIASES.get(cleaned, cleaned)


def to_yahoo_ticker(t: str) -> str:
    return normalize_ticker(t)


def is_fund(row: dict | pd.Series) -> bool:
    """Return True for ETFs and mutual funds."""
    if isinstance(row, pd.Series):
        row = row.to_dict()
    quote_type = str(row.get("quoteType") or "").upper()
    if quote_type in {"ETF", "MUTUALFUND"}:
        return True
    ticker = normalize_ticker(str(row.get("ticker") or ""))
    return ticker in KNOWN_ETF_TICKERS


def effective_price(row: dict | pd.Series) -> float | None:
    """Return best available price from a fundamentals row."""
    if isinstance(row, pd.Series):
        row = row.to_dict()
    price = row.get("currentPrice")
    if price is None or pd.isna(price):
        price = row.get("regularMarketPrice")
    if price is None or pd.isna(price):
        return None
    try:
        parsed = float(price)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _retry_once(fn: Callable[..., T], *args, **kwargs) -> T:
    """Try once, retry after a short sleep on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as first_exc:
        logger.warning("Retrying after error: %s", first_exc)
        time.sleep(1)
        return fn(*args, **kwargs)


def _fetch_ticker_info(ticker: str) -> dict:
    return yf.Ticker(to_yahoo_ticker(ticker)).info or {}


def _normalize_debt_to_equity(value: Any) -> Any:
    """Convert yfinance percentage-scale D/E (e.g. 600) to ratio (6.0)."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    if num > 10:
        return num / 100.0
    return num


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """Pull fundamental metrics for each ticker; skip failures individually."""
    rows: dict[str, dict] = {}

    for ticker in tickers:
        normalized = normalize_ticker(ticker)
        try:
            info = _retry_once(_fetch_ticker_info, normalized)
            row = {field: info.get(field) for field in FUNDAMENTAL_FIELDS}
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            row["currentPrice"] = current_price
            row["regularMarketPrice"] = info.get("regularMarketPrice")
            row["ticker"] = normalized
            if row.get("debtToEquity") is not None:
                row["debtToEquity"] = _normalize_debt_to_equity(row["debtToEquity"])
            rows[normalized] = row
        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", normalized, exc)

    if not rows:
        return pd.DataFrame(columns=["ticker", *FUNDAMENTAL_FIELDS])

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reset_index(drop=True)


def _download_prices(ticker_list: list[str], period: str) -> pd.DataFrame:
    yahoo_symbols = [to_yahoo_ticker(t) for t in ticker_list]
    data = yf.download(
        yahoo_symbols,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if data.empty or len(ticker_list) == 1:
        return data

    normalized = [normalize_ticker(t) for t in ticker_list]
    rename_map = {
        yahoo: norm
        for yahoo, norm in zip(yahoo_symbols, normalized)
        if yahoo != norm
    }
    if rename_map and "Close" in data.columns:
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            data["Close"] = close.rename(columns=rename_map)
    return data


def fetch_price_history(
    tickers: Iterable[str],
    period: str = "2y",
) -> pd.DataFrame:
    """Return wide DataFrame of adjusted close prices."""
    ticker_list = [normalize_ticker(t) for t in tickers]
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


def score_momentum_from_prices(price_series: pd.Series) -> float:
    """Score price momentum 0-100 from 50/200-day MAs; neutral if history is thin."""
    series = price_series.dropna()
    if len(series) < MOMENTUM_MIN_HISTORY_DAYS:
        return MOMENTUM_NEUTRAL_SCORE

    current = float(series.iloc[-1])
    ma50 = float(series.tail(50).mean())
    ma200 = float(series.tail(200).mean())
    if ma200 <= 0:
        return MOMENTUM_NEUTRAL_SCORE

    above_200 = current / ma200
    above_50 = current / ma50 if ma50 > 0 else 1.0
    raw = ((above_200 - 0.85) / 0.30) * 0.6 + ((above_50 - 0.95) / 0.10) * 0.4
    return float(max(0.0, min(100.0, raw * 100)))


def compute_momentum_scores(tickers: Iterable[str], period: str = "2y") -> dict[str, float]:
    """Return per-ticker momentum scores; neutral 50 when history is too short."""
    ticker_list = [normalize_ticker(t) for t in tickers]
    if not ticker_list:
        return {}

    prices = fetch_price_history(ticker_list, period=period)
    scores: dict[str, float] = {}
    for ticker in ticker_list:
        if prices.empty or ticker not in prices.columns:
            scores[ticker] = MOMENTUM_NEUTRAL_SCORE
            continue
        scores[ticker] = score_momentum_from_prices(prices[ticker])
    return scores


if __name__ == "__main__":
    df = fetch_fundamentals(["AAPL", "MSFT"])
    print(df[["ticker", "shortName", "sector", "trailingPE", "beta"]])
    assert len(df) == 2, "Expected 2 rows"
    assert "trailingPE" in df.columns, "Missing trailingPE column"

    etfs = fetch_fundamentals(["QQQ", "VOO", "IVV"])
    for ticker in ["QQQ", "VOO", "IVV"]:
        row = etfs[etfs["ticker"] == ticker].iloc[0]
        assert effective_price(row) is not None, f"Missing price for {ticker}"

    assert normalize_ticker("BRKB") == "BRK-B"
    brk = fetch_fundamentals(["BRKB"])
    assert not brk.empty and effective_price(brk.iloc[0]) is not None

    invalid = fetch_fundamentals(["INVALID1", "INVALID2"])
    assert isinstance(invalid, pd.DataFrame)
    print("data_fetcher checkpoint passed")
