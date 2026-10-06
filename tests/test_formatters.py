"""Tests for display and metric formatters."""

from __future__ import annotations

from portfolio_advisor.formatters import (
    format_dividend_yield,
    format_large_usd,
    format_pct,
    format_trade_action,
    sanitize_peg,
)


def test_format_large_usd_billions():
    assert format_large_usd(101307228160) == "$101.3B"


def test_dividend_yield_scale_aware():
    assert format_dividend_yield(0.0043) == "0.43%"
    assert format_dividend_yield(0.43) == "0.43%"
    assert format_dividend_yield(2.03) == "2.03%"


def test_format_pct_fraction():
    assert format_pct(0.066) == "6.6%"


def test_sanitize_peg_flags_unreliable():
    peg, flag = sanitize_peg(10.06, 0.02)
    assert peg == 10.06
    assert flag is not None and "unreliable" in flag


def test_format_trade_action_labels():
    assert format_trade_action(100.0, threshold=50.0) == "Buy $100"
    assert format_trade_action(-100.0, threshold=50.0) == "Sell $100"
    assert format_trade_action(10.0, threshold=50.0) == "Hold"
