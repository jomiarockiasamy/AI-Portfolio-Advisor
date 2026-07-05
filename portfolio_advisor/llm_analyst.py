"""Provider-agnostic LLM investment thesis generation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

REQUIRED_KEYS = [
    "ticker",
    "moat",
    "growth_drivers",
    "key_risks",
    "thesis_summary",
    "conviction",
]

GROQ_MODEL = "llama-3.3-70b-versatile"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
LLM_TIMEOUT_SECONDS = 30


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


def _build_prompt(ticker: str, fundamentals_row: pd.Series | dict) -> str:
    if isinstance(fundamentals_row, pd.Series):
        row = fundamentals_row.to_dict()
    else:
        row = dict(fundamentals_row)

    metrics = {
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

    return f"""You are a equity research analyst. Analyze {ticker} using the fundamentals below.

Fundamentals:
{json.dumps(metrics, indent=2, default=str)}

Return ONLY valid JSON (no markdown, no commentary) with exactly these fields:
- ticker (string)
- moat (string, 1-2 sentences)
- growth_drivers (string, 1-2 sentences)
- key_risks (string, 1-2 sentences)
- thesis_summary (string, 2-3 sentences)
- conviction (one of: high, medium, low)
"""


def _call_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=LLM_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content or ""


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


def generate_thesis(ticker: str, fundamentals_row: pd.Series | dict) -> Dict[str, Any]:
    """Generate a structured investment thesis for one ticker."""
    if not llm_available():
        return {"ticker": ticker, "error": "no_api_key_configured"}

    prompt = _build_prompt(ticker, fundamentals_row)
    try:
        raw = _call_llm(prompt)
        parsed = json.loads(_strip_markdown_fences(raw))
        for key in REQUIRED_KEYS:
            parsed.setdefault(key, "" if key != "ticker" else ticker)
        parsed["ticker"] = ticker
        return parsed
    except TimeoutError as exc:
        return {"ticker": ticker, "error": f"timeout: {exc}"}
    except Exception as exc:
        exc_name = type(exc).__name__
        if "Timeout" in exc_name or exc_name == "ReadTimeout":
            return {"ticker": ticker, "error": f"timeout: {exc}"}
        return {"ticker": ticker, "error": str(exc)}


def generate_theses_for_candidates(
    candidates_df: pd.DataFrame,
    cache_get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    cache_set: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Generate theses for all candidates, using cache when available."""
    results: List[Dict[str, Any]] = []

    for _, row in candidates_df.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue

        if cache_get is not None:
            cached = cache_get(ticker)
            if cached is not None and not cached.get("error"):
                cached = dict(cached)
                cached["_from_cache"] = True
                results.append(cached)
                continue

        thesis = generate_thesis(ticker, row)
        if cache_set is not None and not thesis.get("error"):
            cache_set(ticker, thesis)
        results.append(thesis)

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
    finally:
        for key, value in saved_keys.items():
            if value is not None:
                os.environ[key] = value

    if os.environ.get("GROQ_API_KEY"):
        live = generate_thesis("AAPL", sample)
        assert not live.get("error"), live
        for key in REQUIRED_KEYS:
            assert key in live, f"Missing key: {key}"
        print("Groq live checkpoint passed")
