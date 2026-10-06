"""Provider-agnostic LLM investment thesis generation."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from portfolio_advisor.data_fetcher import is_fund
from portfolio_advisor.formatters import format_fund_fields, format_pct, format_ratio, sanitize_peg
from portfolio_advisor.signal_engine import compute_sector_medians

EQUITY_REQUIRED_KEYS = [
    "ticker",
    "cited_inputs",
    "signal_reasoning",
    "no_news_note",
    "moat",
    "growth_drivers",
    "key_risks",
    "thesis_summary",
]

FUND_REQUIRED_KEYS = [
    "ticker",
    "cited_inputs",
    "signal_reasoning",
    "no_news_note",
    "fund_role",
    "growth_drivers",
    "key_risks",
    "thesis_summary",
]

GROQ_MODEL = "llama-3.3-70b-versatile"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
LLM_TIMEOUT_SECONDS = 30
THESIS_INTER_CALL_DELAY_SEC = 1.0
MAX_BATCH_THESIS_SECONDS = 600
RETRY_BACKOFF_SECONDS = (2.0, 4.0, 8.0)

NO_NEWS_FALLBACK = "no material company-specific news this period"

logger = logging.getLogger(__name__)


def _resolve_provider() -> str:
    forced = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if forced in {"groq", "anthropic"}:
        return forced
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def active_provider() -> str:
    return _resolve_provider()


def llm_available() -> bool:
    return _resolve_provider() != "none"


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def build_relative_context(
    ticker: str,
    fundamentals_row: pd.Series | dict,
    universe_fundamentals: pd.DataFrame | None = None,
    momentum_score: float | None = None,
    portfolio_weight: float | None = None,
) -> dict[str, Any]:
    """Build sector-relative and momentum context for grounded prompts."""
    if isinstance(fundamentals_row, pd.Series):
        row = fundamentals_row.to_dict()
    else:
        row = dict(fundamentals_row)

    sector = str(row.get("sector") or "Unknown")
    medians = compute_sector_medians(universe_fundamentals)
    sector_med = medians.get(sector, {})
    pe = row.get("trailingPE")
    pe_med = sector_med.get("trailingPE")
    peg_med = sector_med.get("pegRatio")

    valuation_notes: list[str] = []
    if pe and pe_med and pe > 0 and pe_med > 0:
        pct = (pe / pe_med - 1.0) * 100
        direction = "above" if pct > 0 else "below"
        valuation_notes.append(
            f"trailing P/E {format_ratio(pe)} is {abs(pct):.0f}% {direction} "
            f"{sector} median {format_ratio(pe_med)}"
        )
    peg_val, peg_flag = sanitize_peg(row.get("pegRatio"), row.get("earningsGrowth"))
    if peg_val and peg_med and peg_med > 0 and not peg_flag:
        pct = (peg_val / peg_med - 1.0) * 100
        direction = "above" if pct > 0 else "below"
        valuation_notes.append(
            f"PEG {format_ratio(peg_val)} is {abs(pct):.0f}% {direction} "
            f"{sector} median {format_ratio(peg_med)}"
        )
    elif peg_flag:
        valuation_notes.append(peg_flag)

    context: dict[str, Any] = {
        "sector": sector,
        "sector_median_trailing_pe": pe_med,
        "sector_median_peg": peg_med,
        "valuation_vs_sector": valuation_notes,
    }
    if momentum_score is not None:
        context["momentum_score"] = round(float(momentum_score), 1)
    if portfolio_weight is not None:
        context["portfolio_weight_pct"] = round(float(portfolio_weight) * 100, 2)
    return context


def _format_enriched_context(enriched_context: dict[str, Any]) -> str:
    lines: list[str] = []

    if enriched_context.get("is_fund"):
        lines.append("Asset type: ETF/fund (no corporate moat or management analysis).")
        fund_fields = enriched_context.get("fund_fields") or {}
        if fund_fields:
            display_fields = format_fund_fields(fund_fields)
            lines.append(f"Fund metrics: {json.dumps(display_fields, default=str)}")

    ratings = enriched_context.get("analyst_ratings") or {}
    if ratings:
        lines.append(
            "Analyst ratings: "
            f"strong_buy={ratings.get('strong_buy', 0)}, "
            f"buy={ratings.get('buy', 0)}, "
            f"hold={ratings.get('hold', 0)}, "
            f"sell={ratings.get('sell', 0)}, "
            f"strong_sell={ratings.get('strong_sell', 0)}"
        )

    news_status = enriched_context.get("news_status", "ok")
    news = enriched_context.get("recent_news") or []
    if news:
        lines.append("Filtered company-specific news:")
        for item in news[:5]:
            headline = item.get("headline", "")
            published = item.get("datetime") or "unknown date"
            category = item.get("category", "other")
            summary = str(item.get("summary") or "")[:180]
            lines.append(f"- [{published}] ({category}) {headline}: {summary}")
    else:
        lines.append(f"News status: {news_status} (no company-specific headlines passed filter).")

    earnings = enriched_context.get("earnings") or {}
    if earnings:
        parts = []
        if earnings.get("next_date"):
            parts.append(f"date={earnings['next_date']}")
        if earnings.get("eps_estimate") is not None:
            parts.append(f"EPS estimate={earnings['eps_estimate']}")
        if earnings.get("prior_actual_eps") is not None:
            parts.append(f"prior actual EPS={earnings['prior_actual_eps']}")
        if parts:
            lines.append("Upcoming earnings: " + ", ".join(parts))

    return "\n".join(lines)


def _format_relative_context(relative_context: dict[str, Any] | None) -> str:
    if not relative_context:
        return ""
    return json.dumps(relative_context, indent=2, default=str)


def _format_comparative_context(comparative_context: list[dict] | None) -> str:
    if not comparative_context:
        return ""
    return json.dumps(comparative_context, indent=2, default=str)


def _build_prompt(
    ticker: str,
    fundamentals_row: pd.Series | dict,
    enriched_context: Optional[dict[str, Any]] = None,
    relative_context: Optional[dict[str, Any]] = None,
    comparative_context: Optional[list[dict]] = None,
) -> str:
    if isinstance(fundamentals_row, pd.Series):
        row = fundamentals_row.to_dict()
    else:
        row = dict(fundamentals_row)

    fund = bool((enriched_context or {}).get("is_fund")) or is_fund(row)

    equity_metrics = {
        key: row.get(key)
        for key in [
            "shortName",
            "sector",
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
        ]
    }
    fund_metrics = {
        key: row.get(key)
        for key in [
            "shortName",
            "sector",
            "beta",
            "dividendYield",
            "currentPrice",
            "annualReportExpenseRatio",
            "totalAssets",
            "fundFamily",
            "quoteType",
        ]
    }
    metrics = fund_metrics if fund else equity_metrics

    enrichment_block = ""
    if enriched_context:
        formatted = _format_enriched_context(enriched_context)
        if formatted:
            enrichment_block = f"\n\nAdditional context:\n{formatted}\n"

    relative_block = ""
    if relative_context:
        relative_block = f"\n\nRelative context:\n{_format_relative_context(relative_context)}\n"

    comparative_block = ""
    vs_field = ""
    if comparative_context:
        comparative_block = (
            f"\n\nSector alternatives (for comparison):\n"
            f"{_format_comparative_context(comparative_context)}\n"
        )
        vs_field = "\n- vs_alternatives (string, one sentence explaining why this ticker was selected over named alternatives using specific metrics)"

    news_status = (enriched_context or {}).get("news_status", "insufficient")
    has_news = bool((enriched_context or {}).get("recent_news"))

    if fund:
        grounding_rules = f"""
Fund grounding rules:
- Must cite at least 2 fund-relevant metrics from the data (expense ratio, AUM/totalAssets, dividend yield, beta, index name from shortName). Use human-readable formats ($XB, X%).
- Do NOT use moat, management quality, or competitive-advantage language — funds do not have these.
- Do NOT force P/E, profit margins, or debt/equity analysis.
- yfinance debtToEquity when present on underlying data may use percentage scale — do not misread.
- If news_status is insufficient, set no_news_note to "{NO_NEWS_FALLBACK}" and do not invent news.
"""
        json_fields = """
- ticker (string)
- cited_inputs (array of strings — dated facts/metrics/headlines you actually used)
- signal_reasoning (string — must reference at least one cited input)
- no_news_note (string — empty if news exists, else "no material company-specific news this period")
- fund_role (string, 2-3 sentences on index exposure, cost, portfolio role)
- growth_drivers (string, 1-2 sentences)
- key_risks (string, 1-2 sentences)
- thesis_summary (string, 2-3 sentences)
"""
    else:
        grounding_rules = f"""
Equity grounding rules:
- cited_inputs must list specific dated facts (headline + date and/or numeric fundamentals).
- Must cite at least 2 specific numbers from equity fundamentals in cited_inputs and narrative. Use human-readable formats ($XB, X%).
- signal_reasoning must reference at least one item from cited_inputs.
- When recent_news is non-empty, reference at least one headline by exact name and date.
- Forbidden: generic macro boilerplate (Fed, rate hikes, market volatility) unless a provided headline explicitly ties it to this company.
- Do not treat flagged unreliable PEG values as valuation signals.
- yfinance debtToEquity may be percentage-scale (e.g. 600 means 6.0x) — verify before citing as "600x debt".
- If news_status is "{news_status}" and no headlines provided, set no_news_note to "{NO_NEWS_FALLBACK}" and do not invent news or generic commentary to fill space.
"""
        json_fields = """
- ticker (string)
- cited_inputs (array of strings — dated facts/headlines/metrics you actually used)
- signal_reasoning (string — must reference at least one cited input)
- no_news_note (string — empty if news exists, else "no material company-specific news this period")
- moat (string, 1-2 sentences)
- growth_drivers (string, 1-2 sentences)
- key_risks (string, 1-2 sentences)
- thesis_summary (string, 2-3 sentences)
"""

    asset_label = "ETF/fund" if fund else "equity"
    return f"""You are an equity research analyst. Analyze {ticker} ({asset_label}) using the data below.

Fundamentals:
{json.dumps(metrics, indent=2, default=str)}
{enrichment_block}{relative_block}{comparative_block}
{grounding_rules}
Return ONLY valid JSON (no markdown, no commentary) with exactly these fields:
{json_fields}{vs_field}
Do NOT include a conviction field.
News available: {has_news}
"""


def _call_groq(prompt: str) -> str:
    from groq import Groq

    last_exc: Exception | None = None
    for attempt, delay in enumerate(RETRY_BACKOFF_SECONDS):
        try:
            client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=LLM_TIMEOUT_SECONDS)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            exc_name = type(exc).__name__
            msg = str(exc).lower()
            retriable = (
                "429" in msg
                or "rate" in msg
                or "timeout" in exc_name.lower()
                or "connection" in msg
            )
            if retriable and attempt < len(RETRY_BACKOFF_SECONDS) - 1:
                time.sleep(delay)
                continue
            raise
    if last_exc:
        raise last_exc
    return ""


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        timeout=LLM_TIMEOUT_SECONDS,
    )
    parts = [block.text for block in response.content if hasattr(block, "text")]
    return "".join(parts)


def _call_llm(prompt: str) -> str:
    provider = _resolve_provider()
    if provider == "groq":
        return _call_groq(prompt)
    if provider == "anthropic":
        return _call_anthropic(prompt)
    raise RuntimeError("no_api_key_configured")


def _required_keys_for_row(
    fundamentals_row: pd.Series | dict,
    enriched_context: Optional[dict[str, Any]] = None,
    comparative_context: Optional[list[dict]] = None,
) -> list[str]:
    if isinstance(fundamentals_row, pd.Series):
        row = fundamentals_row.to_dict()
    else:
        row = dict(fundamentals_row)
    fund = bool((enriched_context or {}).get("is_fund")) or is_fund(row)
    keys = FUND_REQUIRED_KEYS if fund else EQUITY_REQUIRED_KEYS
    if comparative_context:
        keys = keys + ["vs_alternatives"]
    return keys


def _thesis_is_complete(parsed: dict[str, Any], required_keys: list[str], is_fund: bool) -> bool:
    summary = str(parsed.get("thesis_summary") or "").strip()
    reasoning = str(parsed.get("signal_reasoning") or "").strip()
    if not summary and not reasoning:
        return False

    cited = parsed.get("cited_inputs")
    has_cited = isinstance(cited, list) and any(str(item).strip() for item in cited)
    no_news = str(parsed.get("no_news_note") or "").strip()
    if not has_cited and not no_news:
        return False

    if is_fund and not str(parsed.get("fund_role") or "").strip():
        return False

    return True


def thesis_is_complete(
    thesis: dict[str, Any],
    *,
    is_fund: bool = False,
    enriched_context: Optional[dict[str, Any]] = None,
    fundamentals_row: pd.Series | dict | None = None,
) -> bool:
    """Public completeness check for UI and batch validation."""
    if thesis.get("error"):
        return False
    is_fund_flag = is_fund
    if enriched_context and enriched_context.get("is_fund"):
        is_fund_flag = True
    elif fundamentals_row is not None:
        if isinstance(fundamentals_row, pd.Series):
            row = fundamentals_row.to_dict()
        else:
            row = dict(fundamentals_row)
        is_fund_flag = is_fund_flag or is_fund(row)
    required = FUND_REQUIRED_KEYS if is_fund_flag else EQUITY_REQUIRED_KEYS
    return _thesis_is_complete(thesis, required, is_fund_flag)


def _thesis_narrative_empty(thesis: dict[str, Any]) -> bool:
    fields = ["thesis_summary", "signal_reasoning", "growth_drivers", "key_risks"]
    if thesis.get("fund_role") is not None:
        fields.append("fund_role")
    if thesis.get("moat") is not None:
        fields.append("moat")
    return not any(str(thesis.get(field) or "").strip() for field in fields)


def generate_thesis(
    ticker: str,
    fundamentals_row: pd.Series | dict,
    enriched_context: Optional[dict[str, Any]] = None,
    relative_context: Optional[dict[str, Any]] = None,
    comparative_context: Optional[list[dict]] = None,
) -> Dict[str, Any]:
    """Generate a structured investment thesis for one ticker."""
    if not llm_available():
        return {"ticker": ticker, "error": "no_api_key_configured"}

    prompt = _build_prompt(
        ticker,
        fundamentals_row,
        enriched_context,
        relative_context=relative_context,
        comparative_context=comparative_context,
    )
    required_keys = _required_keys_for_row(
        fundamentals_row, enriched_context, comparative_context
    )
    try:
        raw = _call_llm(prompt)
        parsed = json.loads(_strip_markdown_fences(raw))
        for key in required_keys:
            if key == "cited_inputs" and not isinstance(parsed.get(key), list):
                parsed[key] = parsed.get(key, []) if parsed.get(key) else []
            else:
                parsed.setdefault(key, "" if key != "ticker" else ticker)
        parsed["ticker"] = ticker
        parsed.pop("conviction", None)
        fund = bool((enriched_context or {}).get("is_fund")) or is_fund(
            fundamentals_row.to_dict()
            if isinstance(fundamentals_row, pd.Series)
            else dict(fundamentals_row)
        )
        if not _thesis_is_complete(parsed, required_keys, fund):
            return {"ticker": ticker, "error": "incomplete_thesis: empty LLM response"}
        return parsed
    except TimeoutError as exc:
        return {"ticker": ticker, "error": f"timeout: {exc}"}
    except Exception as exc:
        exc_name = type(exc).__name__
        if "Timeout" in exc_name or exc_name == "ReadTimeout":
            return {"ticker": ticker, "error": f"timeout: {exc}"}
        return {"ticker": ticker, "error": str(exc)}


def score_news_sentiment(
    ticker: str,
    headlines: list[str],
    cache_get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    cache_set: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Score recent headlines from -1 (negative) to +1 (positive)."""
    cleaned = [h.strip() for h in headlines if h and str(h).strip()]
    if len(cleaned) < 2:
        return {
            "ticker": ticker,
            "score": 0.0,
            "justification": "Insufficient news data",
            "insufficient_data": True,
        }

    if cache_get is not None:
        cached = cache_get(ticker)
        if cached is not None:
            return cached

    if not llm_available():
        return {
            "ticker": ticker,
            "score": 0.0,
            "justification": "LLM not configured",
            "insufficient_data": True,
        }

    prompt = f"""Score the overall sentiment of these recent headlines for {ticker}.
Headlines:
{json.dumps(cleaned[:5], indent=2)}

Return ONLY valid JSON with:
- score (number from -1 to 1)
- justification (one short sentence)
"""
    try:
        raw = _call_llm(prompt)
        parsed = json.loads(_strip_markdown_fences(raw))
        score = float(parsed.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        result = {
            "ticker": ticker,
            "score": score,
            "justification": str(parsed.get("justification", "")),
            "insufficient_data": False,
        }
        if cache_set is not None:
            cache_set(ticker, result)
        return result
    except Exception as exc:
        return {
            "ticker": ticker,
            "score": 0.0,
            "justification": str(exc),
            "insufficient_data": True,
        }


def score_news_sentiments_for_tickers(
    enriched_by_ticker: dict[str, dict],
    cache_get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    cache_set: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> dict[str, dict]:
    """Score news sentiment for each ticker with Finnhub headlines."""
    results: dict[str, dict] = {}
    for ticker, context in enriched_by_ticker.items():
        news = context.get("recent_news") or []
        headlines = [item.get("headline", "") for item in news if isinstance(item, dict)]
        results[ticker.upper()] = score_news_sentiment(
            ticker, headlines, cache_get=cache_get, cache_set=cache_set
        )
    return results


def generate_theses_for_candidates(
    candidates_df: pd.DataFrame,
    cache_get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    cache_set: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    enriched_by_ticker: Optional[dict[str, dict[str, Any]]] = None,
    relative_context_by_ticker: Optional[dict[str, dict[str, Any]]] = None,
    comparative_context_by_ticker: Optional[dict[str, list[dict]]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    tickers_filter: Optional[list[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate theses for all candidates, using cache when available."""
    results: List[Dict[str, Any]] = []
    use_cache = enriched_by_ticker is None and relative_context_by_ticker is None
    filter_set = {t.upper() for t in tickers_filter} if tickers_filter else None

    rows: list[pd.Series] = []
    for _, row in candidates_df.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        if filter_set and ticker not in filter_set:
            continue
        rows.append(row)

    total = len(rows)
    batch_start = time.time()

    for idx, row in enumerate(rows, start=1):
        if time.time() - batch_start > MAX_BATCH_THESIS_SECONDS:
            remaining = rows[idx - 1 :]
            for pending in remaining:
                pending_ticker = str(pending.get("ticker", "")).upper()
                if on_progress:
                    on_progress(idx, total, pending_ticker)
                err = {
                    "ticker": pending_ticker,
                    "error": "batch_time_budget_exceeded",
                }
                results.append(err)
                print(f"[thesis] {pending_ticker}: FAILED — batch_time_budget_exceeded")
            break

        ticker = str(row.get("ticker", "")).upper()
        if on_progress:
            on_progress(idx, total, ticker)

        enrichment = (enriched_by_ticker or {}).get(ticker)
        relative = (relative_context_by_ticker or {}).get(ticker)
        comparative = (comparative_context_by_ticker or {}).get(ticker)

        if use_cache and cache_get is not None:
            cached = cache_get(ticker)
            if cached is not None and not cached.get("error"):
                fund = bool((enrichment or {}).get("is_fund")) or is_fund(row.to_dict())
                if _thesis_is_complete(
                    cached,
                    _required_keys_for_row(row, enrichment, comparative),
                    fund,
                ):
                    cached = dict(cached)
                    cached["_from_cache"] = True
                    results.append(cached)
                    print(f"[thesis] {ticker}: ok (cached)")
                    if idx < total:
                        time.sleep(THESIS_INTER_CALL_DELAY_SEC)
                    continue

        thesis = generate_thesis(
            ticker,
            row,
            enrichment,
            relative_context=relative,
            comparative_context=comparative,
        )
        if thesis.get("error"):
            print(f"[thesis] {ticker}: FAILED — {thesis['error']}")
        else:
            print(f"[thesis] {ticker}: ok")
            if use_cache and cache_set is not None:
                cache_set(ticker, thesis)
        results.append(thesis)

        if idx < total:
            time.sleep(THESIS_INTER_CALL_DELAY_SEC)

    return results


if __name__ == "__main__":
    sample = pd.Series(
        {
            "ticker": "AAPL",
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "trailingPE": 28.0,
            "beta": 1.2,
        }
    )

    saved_keys = {
        k: os.environ.pop(k, None)
        for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER")
    }
    try:
        result = generate_thesis("AAPL", sample)
        assert result.get("error") == "no_api_key_configured"
        print("No-key checkpoint passed")

        prompt = _build_prompt("AAPL", sample, {"news_status": "insufficient", "recent_news": []})
        assert "cited_inputs" in prompt
        assert "conviction" not in prompt or "Do NOT include a conviction field" in prompt
        print("Prompt schema checkpoint passed")
    finally:
        for key, value in saved_keys.items():
            if value is not None:
                os.environ[key] = value

    if os.environ.get("GROQ_API_KEY"):
        live = generate_thesis("AAPL", sample)
        assert not live.get("error"), live
        for key in EQUITY_REQUIRED_KEYS:
            assert key in live, f"Missing key: {key}"
        assert "conviction" not in live
        print("Groq live checkpoint passed")
