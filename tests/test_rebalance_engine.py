"""Tests for signal-aligned rebalance tilt and cap enforcement."""

from __future__ import annotations

from portfolio_advisor.formatters import snap_weight_dust
from portfolio_advisor.rebalance_engine import (
    BUY_MIN_WEIGHT,
    _enforce_weight_cap,
    _tilt_single_weight,
    apply_signal_tilt,
)


def test_snap_weight_dust():
    assert snap_weight_dust(0.0000000000000006) == 0.0


def test_hold_floor_when_optimizer_zeros_position():
    hold_signal = {"signal": "HOLD", "composite_score": 50.0}
    tilted = _tilt_single_weight(0.0, 0.27, hold_signal)
    assert tilted >= 0.135 - 1e-9


def test_buy_floor_when_base_weight_zero():
    buy_signal = {"signal": "BUY", "composite_score": 67.0}
    tilted = _tilt_single_weight(0.0, 0.0, buy_signal)
    assert tilted >= BUY_MIN_WEIGHT - 1e-9


def test_cap_enforcement_no_weight_above_max():
    naive = {f"T{i}": 0.14 for i in range(6)}
    naive["T6"] = 0.16
    enforced = _enforce_weight_cap(naive, 0.15)
    assert all(w <= 0.15 + 1e-6 for w in enforced.values())
    assert abs(sum(enforced.values()) - 1.0) < 1e-6


def test_apply_signal_tilt_sums_to_one_and_respects_cap():
    base = {"A": 0.14, "B": 0.14, "C": 0.14, "D": 0.14, "E": 0.14, "F": 0.14, "G": 0.16}
    current = {k: 0.1 for k in base}
    current["G"] = 0.4
    signals = {
        "A": {"signal": "BUY", "composite_score": 70.0},
        "B": {"signal": "HOLD", "composite_score": 55.0},
        "C": {"signal": "SELL", "composite_score": 35.0},
        "D": {"signal": "HOLD", "composite_score": 50.0},
        "E": {"signal": "BUY", "composite_score": 68.0},
        "F": {"signal": "HOLD", "composite_score": 52.0},
        "G": {"signal": "SELL", "composite_score": 38.0},
    }
    result = apply_signal_tilt(base, current, signals, 0.15)
    assert abs(sum(result.values()) - 1.0) < 1e-6
    assert all(w <= 0.15 + 1e-6 for w in result.values())
