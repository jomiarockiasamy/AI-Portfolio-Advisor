"""Human-readable formatting for fundamentals and cited metrics."""

from __future__ import annotations

import math
from typing import Any


def format_large_usd(value: Any) -> str:
    """Format dollar amounts as $101.3B, $888.1M, etc."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(num) or math.isinf(num):
        return str(value)
    sign = "-" if num < 0 else ""
    num = abs(num)
    if num >= 1_000_000_000_000:
        return f"{sign}${num / 1_000_000_000_000:.1f}T"
    if num >= 1_000_000_000:
        return f"{sign}${num / 1_000_000_000:.1f}B"
    if num >= 1_000_000:
        return f"{sign}${num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{sign}${num / 1_000:.1f}K"
    return f"{sign}${num:,.2f}"


def format_dividend_yield(value: Any) -> str:
    """Scale-aware dividend yield — yfinance may use fraction or percent points."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(num) or math.isinf(num):
        return str(value)
    if num <= 0.05:
        return f"{num * 100:.2f}%"
    return f"{num:.2f}%"


def format_pct(value: Any, decimals: int = 1) -> str:
    """Format fractional rates (e.g. revenueGrowth 0.066) as percentages."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(num) or math.isinf(num):
        return str(value)
    if abs(num) <= 1.0:
        return f"{num * 100:.{decimals}f}%"
    return f"{num:.{decimals}f}%"


def format_weight_pct(weight: float, decimals: int = 1) -> str:
    return f"{weight * 100:.{decimals}f}%"


WEIGHT_DUST_THRESHOLD = 0.001


def snap_weight_dust(weight: float, threshold: float = WEIGHT_DUST_THRESHOLD) -> float:
    if abs(weight) < threshold:
        return 0.0
    return weight


def snap_weights_dict(
    weights: dict[str, float],
    threshold: float = WEIGHT_DUST_THRESHOLD,
) -> dict[str, float]:
    snapped = {ticker: snap_weight_dust(float(w), threshold) for ticker, w in weights.items()}
    total = sum(snapped.values())
    if total <= 0:
        return snapped
    if abs(total - 1.0) > 1e-9:
        return {ticker: w / total for ticker, w in snapped.items()}
    return snapped


def format_trade_action(delta_value: float, threshold: float = 50.0) -> str:
    if abs(delta_value) < threshold:
        return "Hold"
    if delta_value > 0:
        return f"Buy ${delta_value:,.0f}"
    return f"Sell ${abs(delta_value):,.0f}"


def format_ratio(value: Any, decimals: int = 2) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(num) or math.isinf(num):
        return str(value)
    return f"{num:.{decimals}f}"


def sanitize_peg(
    peg: Any,
    earnings_growth: Any,
    peg_max: float = 5.0,
    min_growth: float = 0.05,
) -> tuple[float | None, str | None]:
    """Return (peg, flag) where flag describes unreliable PEG inputs."""
    try:
        peg_val = float(peg)
    except (TypeError, ValueError):
        return None, None
    if math.isnan(peg_val) or peg_val <= 0:
        return None, None

    growth_val = None
    try:
        if earnings_growth is not None:
            growth_val = float(earnings_growth)
            if math.isnan(growth_val):
                growth_val = None
    except (TypeError, ValueError):
        growth_val = None

    unreliable = peg_val > peg_max or (
        growth_val is not None and abs(growth_val) < min_growth
    )
    if unreliable:
        return peg_val, (
            f"PEG {peg_val:.2f} (unreliable — low or noisy earnings growth denominator)"
        )
    return peg_val, None


def format_fund_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Return display-friendly fund metric dict for prompts."""
    out: dict[str, Any] = {}
    for key, val in fields.items():
        if val is None:
            continue
        if key == "totalAssets":
            out[key] = format_large_usd(val)
        elif key == "dividendYield":
            out[key] = format_dividend_yield(val)
        elif key == "annualReportExpenseRatio":
            out[key] = format_pct(val, decimals=2)
        elif key in {"beta", "shortName", "fundFamily"}:
            out[key] = val
        else:
            out[key] = val
    return out


if __name__ == "__main__":
    assert format_large_usd(101307228160) == "$101.3B"
    assert format_large_usd(888128929792) == "$888.1B"
    assert format_dividend_yield(0.0043) == "0.43%"
    assert format_dividend_yield(0.43) == "0.43%"
    assert format_dividend_yield(2.03) == "2.03%"
    assert format_pct(0.066) == "6.6%"
    assert snap_weight_dust(0.0000000000000006) == 0.0

    peg, flag = sanitize_peg(10.06, 0.02)
    assert peg == 10.06
    assert flag is not None and "unreliable" in flag

    peg_ok, flag_ok = sanitize_peg(1.2, 0.15)
    assert peg_ok == 1.2
    assert flag_ok is None

    print("formatters checkpoint passed")
