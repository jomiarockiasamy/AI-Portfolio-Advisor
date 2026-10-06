"""Signal-aligned rebalance tilt and weight-cap enforcement."""

from __future__ import annotations

from typing import Any

from portfolio_advisor.formatters import snap_weight_dust, snap_weights_dict
from portfolio_advisor.optimizer import RISK_PARAMS
from portfolio_advisor.signal_engine import BUY_THRESHOLD, SELL_THRESHOLD

WEIGHT_DUST_THRESHOLD = 0.001
BUY_MIN_WEIGHT = 0.03

REBALANCE_MODES = ("math_only", "signal_aligned")

REBALANCE_MODE_LABELS: dict[str, str] = {
    "math_only": "Math-only (price history)",
    "signal_aligned": "Signal-aligned (optimizer + BUY/HOLD/SELL)",
}

REBALANCE_MODE_DESCRIPTIONS: dict[str, str] = {
    "math_only": (
        "Targets come from historical return/risk optimization only. "
        "Thesis signals are separate."
    ),
    "signal_aligned": (
        "Targets start from the optimizer, then are adjusted by composite "
        "BUY/HOLD/SELL scores."
    ),
}


def get_max_weight(risk_tolerance: str) -> float:
    return RISK_PARAMS.get(risk_tolerance, RISK_PARAMS["moderate"])["max_weight"]


def _enforce_weight_cap(
    weights: dict[str, float],
    max_weight: float,
    tol: float = 1e-9,
    max_iter: int = 50,
) -> dict[str, float]:
    """Iteratively clip and redistribute until weights sum to 1 and respect max_weight."""
    w = {ticker: max(0.0, float(value)) for ticker, value in weights.items()}
    if not w:
        return w

    for _ in range(max_iter):
        w = {ticker: min(max_weight, value) for ticker, value in w.items()}
        total = sum(w.values())
        if total <= tol:
            return {ticker: 0.0 for ticker in w}

        capped = all(value <= max_weight + tol for value in w.values())
        if capped and abs(total - 1.0) < tol:
            return w

        if total < 1.0 - tol:
            shortfall = 1.0 - total
            room = {ticker: max_weight - w[ticker] for ticker in w if w[ticker] < max_weight - tol}
            room_total = sum(room.values())
            if room_total <= tol:
                break
            for ticker, headroom in room.items():
                w[ticker] += shortfall * (headroom / room_total)
        else:
            w = {ticker: value / total for ticker, value in w.items()}

    w = {ticker: min(max_weight, value) for ticker, value in w.items()}
    total = sum(w.values())
    if total > tol:
        w = {ticker: value / total for ticker, value in w.items()}
    return w


def _tilt_single_weight(
    base: float,
    current: float,
    signal: dict[str, Any],
) -> float:
    composite = float(signal.get("composite_score", 50.0))
    label = str(signal.get("signal", "HOLD"))

    if label == "BUY" or composite >= BUY_THRESHOLD:
        multiplier = 1.0 + (composite - BUY_THRESHOLD) / 100.0
        tilted = base * multiplier
        tilted = max(tilted, BUY_MIN_WEIGHT)
        if current > 0:
            tilted = max(tilted, current * 0.25)
        return tilted

    if label == "SELL" or composite <= SELL_THRESHOLD:
        multiplier = max(0.1, composite / SELL_THRESHOLD) if SELL_THRESHOLD > 0 else 0.1
        return base * multiplier

    return max(base, current * 0.5)


def apply_signal_tilt(
    base_weights: dict[str, float],
    current_weights: dict[str, float],
    signals: dict[str, dict[str, Any]],
    max_weight: float,
) -> dict[str, float]:
    """Apply BUY/HOLD/SELL tilt to optimizer weights, then enforce cap iteratively."""
    tickers = set(base_weights) | set(current_weights) | set(signals)
    tilted: dict[str, float] = {}
    for ticker in tickers:
        base = float(base_weights.get(ticker, 0.0))
        current = float(current_weights.get(ticker, 0.0))
        signal = signals.get(ticker, {})
        tilted[ticker] = _tilt_single_weight(base, current, signal)

    total = sum(tilted.values())
    if total > 0:
        tilted = {ticker: value / total for ticker, value in tilted.items()}

    capped = _enforce_weight_cap(tilted, max_weight)
    return snap_weights_dict(capped)


if __name__ == "__main__":
    from portfolio_advisor.formatters import snap_weight_dust

    assert snap_weight_dust(0.0000000000000006) == 0.0

    hold_signal = {"signal": "HOLD", "composite_score": 50.0}
    hold_tilted = _tilt_single_weight(0.0, 0.27, hold_signal)
    assert hold_tilted >= 0.135 - 1e-9

    buy_signal = {"signal": "BUY", "composite_score": 67.0}
    buy_tilted = _tilt_single_weight(0.0, 0.0, buy_signal)
    assert buy_tilted >= BUY_MIN_WEIGHT - 1e-9

    sell_signal = {"signal": "SELL", "composite_score": 30.0}
    assert _tilt_single_weight(0.2, 0.1, sell_signal) < 0.2

    base = {"A": 0.14, "B": 0.14, "C": 0.14, "D": 0.14, "E": 0.14, "F": 0.14, "G": 0.16}
    current = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1, "F": 0.1, "G": 0.4}
    signals = {
        "A": {"signal": "BUY", "composite_score": 70.0},
        "B": {"signal": "HOLD", "composite_score": 55.0},
        "C": {"signal": "SELL", "composite_score": 35.0},
        "D": {"signal": "HOLD", "composite_score": 50.0},
        "E": {"signal": "BUY", "composite_score": 68.0},
        "F": {"signal": "HOLD", "composite_score": 52.0},
        "G": {"signal": "SELL", "composite_score": 38.0},
    }
    max_w = 0.15
    result = apply_signal_tilt(base, current, signals, max_w)
    assert abs(sum(result.values()) - 1.0) < 1e-6
    assert all(w <= max_w + 1e-6 for w in result.values())
    assert result.get("A", 0) >= BUY_MIN_WEIGHT - 1e-6

    naive_break = {"T1": 0.14, "T2": 0.14, "T3": 0.14, "T4": 0.14, "T5": 0.14, "T6": 0.14, "T7": 0.16}
    enforced = _enforce_weight_cap(naive_break, 0.15)
    assert all(w <= 0.15 + 1e-6 for w in enforced.values())
    assert abs(sum(enforced.values()) - 1.0) < 1e-6

    print("rebalance_engine checkpoint passed")
