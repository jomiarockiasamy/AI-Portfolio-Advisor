"""Finnhub REST helpers for analyst ratings, news, and earnings enrichment."""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
MAX_CALLS_PER_MINUTE = 60
_RETRY_SLEEP_SECONDS = 2.0
NEWS_LOOKBACK_DAYS = 10

# Occasional upkeep: add aliases when wire services use names not in yfinance shortName.
COMPANY_NAME_ALIASES: dict[str, list[str]] = {
    "BRK-B": ["Berkshire", "Buffett"],
    "META": ["Facebook"],
    "GOOGL": ["Alphabet", "Google"],
    "GOOG": ["Alphabet", "Google"],
}

GENERIC_NEWS_PATTERNS = [
    r"stocks to watch",
    r"market wrap",
    r"stock market today",
    r"wall street",
    r"top \d+ stocks",
    r"best stocks",
]

NEWS_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings": ["earnings", "eps", "quarterly results", "q1", "q2", "q3", "q4"],
    "guidance": ["guidance", "outlook", "forecast", "raises outlook", "lowers outlook"],
    "M&A": ["acquisition", "merger", "takeover", "buyout", "deal to buy"],
    "regulatory": ["fda", "sec ", "regulator", "antitrust", "investigation", "lawsuit"],
    "executive_change": ["ceo", "cfo", "executive", "resign", "appoints", "steps down"],
    "analyst_rating": ["upgrade", "downgrade", "price target", "analyst", "rating"],
    "product_launch": ["launch", "unveil", "introduces", "new product", "release"],
}

_call_timestamps: list[float] = []


def finnhub_configured() -> bool:
    """Return True when FINNHUB_API_KEY is set."""
    return bool(os.environ.get("FINNHUB_API_KEY", "").strip())


def _throttle() -> None:
    global _call_timestamps
    now = time.time()
    _call_timestamps = [ts for ts in _call_timestamps if now - ts < 60.0]
    if len(_call_timestamps) >= MAX_CALLS_PER_MINUTE:
        sleep_for = 60.0 - (now - _call_timestamps[0]) + 0.05
        if sleep_for > 0:
            time.sleep(sleep_for)
        now = time.time()
        _call_timestamps = [ts for ts in _call_timestamps if now - ts < 60.0]
    _call_timestamps.append(time.time())


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return None

    query = dict(params or {})
    query["token"] = api_key

    for attempt in range(2):
        _throttle()
        try:
            response = requests.get(
                f"{FINNHUB_BASE}{path}",
                params=query,
                timeout=15,
            )
        except requests.RequestException:
            return None

        if response.status_code == 429 and attempt == 0:
            time.sleep(_RETRY_SLEEP_SECONDS)
            continue
        if response.status_code != 200:
            return None

        try:
            return response.json()
        except ValueError:
            return None

    return None


def _normalize_headline(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _headline_key(text: str) -> str:
    words = _normalize_headline(text).split()
    return " ".join(words[:6])


def _word_set(text: str) -> set[str]:
    return {w for w in _normalize_headline(text).split() if len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _derive_name_tokens(short_name: str | None) -> list[str]:
    if not short_name:
        return []
    name = str(short_name)
    for suffix in (
        ", Inc.",
        " Inc.",
        " Corp.",
        " Corporation",
        " Ltd.",
        " PLC",
        " ETF",
        " Trust",
        " Fund",
    ):
        name = name.replace(suffix, "")
    tokens = [t.strip() for t in re.split(r"[\s,&]+", name) if len(t.strip()) >= 3]
    return tokens[:5]


def build_match_tokens(ticker: str, short_name: str | None = None) -> list[str]:
    """Return tokens used to match company-specific news headlines."""
    symbol = ticker.upper()
    tokens = [symbol]
    tokens.extend(COMPANY_NAME_ALIASES.get(symbol, []))
    tokens.extend(_derive_name_tokens(short_name))
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            unique.append(token)
    return unique


def _is_generic_roundup(headline: str, summary: str, match_tokens: list[str]) -> bool:
    text = f"{headline} {summary[:200]}".lower()
    for pattern in GENERIC_NEWS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            company_mentioned = any(token.lower() in text for token in match_tokens)
            if not company_mentioned:
                return True
    if "fed rate" in text or "rate hike" in text:
        company_mentioned = any(token.lower() in text for token in match_tokens)
        if not company_mentioned:
            return True
    return False


def _matches_company(headline: str, summary: str, match_tokens: list[str]) -> bool:
    text = f"{headline} {summary[:200]}".lower()
    for token in match_tokens:
        if token.lower() in text:
            return True
    return False


def categorize_news_item(headline: str, summary: str) -> str:
    """Rule-based news category tagging (no LLM call)."""
    text = f"{headline} {summary}".lower()
    for category, keywords in NEWS_CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return "other"


def dedupe_news_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop near-duplicate headlines; keep the most complete summary."""
    if not items:
        return []
    sorted_items = sorted(
        items,
        key=lambda x: len(str(x.get("summary") or "")),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for item in sorted_items:
        headline = str(item.get("headline") or "")
        key = _headline_key(headline)
        words = _word_set(headline)
        duplicate = False
        for existing in kept:
            if _headline_key(str(existing.get("headline") or "")) == key:
                duplicate = True
                break
            if _jaccard(words, _word_set(str(existing.get("headline") or ""))) >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def filter_company_news(
    raw_news: list[dict[str, Any]],
    ticker: str,
    short_name: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Filter Finnhub news to company-specific items within lookback window."""
    match_tokens = build_match_tokens(ticker, short_name)
    filtered: list[dict[str, Any]] = []

    for item in raw_news:
        headline = str(item.get("headline") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not headline:
            continue
        if _is_generic_roundup(headline, summary, match_tokens):
            continue
        if not _matches_company(headline, summary, match_tokens):
            continue
        tagged = dict(item)
        tagged["category"] = categorize_news_item(headline, summary)
        filtered.append(tagged)

    deduped = dedupe_news_items(filtered)
    status = "ok" if deduped else "insufficient"
    return deduped, status


def get_analyst_ratings(ticker: str) -> dict[str, Any]:
    """Return latest analyst recommendation trend counts for a ticker."""
    from portfolio_advisor.data_fetcher import normalize_ticker

    data = _get("/stock/recommendation", {"symbol": normalize_ticker(ticker)})
    if not isinstance(data, list) or not data:
        return {}

    latest = data[0] if isinstance(data[0], dict) else {}
    return {
        "period": latest.get("period"),
        "strong_buy": latest.get("strongBuy", 0),
        "buy": latest.get("buy", 0),
        "hold": latest.get("hold", 0),
        "sell": latest.get("sell", 0),
        "strong_sell": latest.get("strongSell", 0),
    }


def get_recent_news(
    ticker: str,
    days_back: int = NEWS_LOOKBACK_DAYS,
    short_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return filtered, deduped, categorized company news for a ticker."""
    from portfolio_advisor.data_fetcher import normalize_ticker

    symbol = normalize_ticker(ticker)
    end = date.today()
    start = end - timedelta(days=days_back)
    data = _get(
        "/company-news",
        {
            "symbol": symbol,
            "from": start.isoformat(),
            "to": end.isoformat(),
        },
    )
    if not isinstance(data, list):
        return []

    raw: list[dict[str, Any]] = []
    for item in data[:30]:
        if not isinstance(item, dict):
            continue
        ts = item.get("datetime")
        published = None
        if isinstance(ts, (int, float)):
            published = datetime.utcfromtimestamp(ts).isoformat()
        raw.append(
            {
                "headline": item.get("headline") or "",
                "summary": item.get("summary") or "",
                "datetime": published,
                "url": item.get("url") or "",
            }
        )

    filtered, _ = filter_company_news(raw, symbol, short_name=short_name)
    return filtered


def get_earnings_calendar(ticker: str) -> dict[str, Any]:
    """Return next earnings date and recent EPS estimate vs actual."""
    from portfolio_advisor.data_fetcher import normalize_ticker

    symbol = normalize_ticker(ticker)
    today = date.today()
    window_end = today + timedelta(days=120)

    calendar = _get(
        "/calendar/earnings",
        {"from": today.isoformat(), "to": window_end.isoformat(), "symbol": symbol},
    )
    earnings_history = _get("/stock/earnings", {"symbol": symbol})

    next_date = None
    eps_estimate = None
    if isinstance(calendar, dict):
        for row in calendar.get("earningsCalendar", []) or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol", "")).upper() != symbol:
                continue
            next_date = row.get("date") or row.get("reportDate")
            eps_estimate = row.get("epsEstimate")
            if next_date:
                break

    prior_actual = None
    if isinstance(earnings_history, list) and earnings_history:
        latest = earnings_history[0]
        if isinstance(latest, dict):
            prior_actual = latest.get("actual")

    result: dict[str, Any] = {}
    if next_date:
        result["next_date"] = next_date
    if eps_estimate is not None:
        result["eps_estimate"] = eps_estimate
    if prior_actual is not None:
        result["prior_actual_eps"] = prior_actual
    return result


if __name__ == "__main__":
    sample_raw = [
        {
            "headline": "Apple reports strong iPhone sales",
            "summary": "Apple Inc beat expectations on iPhone revenue.",
            "datetime": "2026-07-20T12:00:00",
        },
        {
            "headline": "Apple reports strong iPhone sales",
            "summary": "Apple Inc beat expectations on iPhone revenue with more detail.",
            "datetime": "2026-07-20T13:00:00",
        },
        {
            "headline": "Fed rate hike fears weigh on stocks to watch",
            "summary": "Markets fell on macro concerns.",
            "datetime": "2026-07-21T12:00:00",
        },
    ]
    filtered, status = filter_company_news(sample_raw, "AAPL", short_name="Apple Inc.")
    assert status == "ok"
    assert len(filtered) == 1
    assert filtered[0]["category"] in {"earnings", "product_launch", "other"}

    generic_only, generic_status = filter_company_news(
        [{"headline": "Top 10 stocks to watch today", "summary": "Market wrap", "datetime": None}],
        "AAPL",
        short_name="Apple Inc.",
    )
    assert generic_status == "insufficient"
    assert generic_only == []

    assert categorize_news_item("Company beats Q2 earnings", "") == "earnings"
    assert dedupe_news_items(sample_raw)
    print("fundamentals_finnhub checkpoint passed")
