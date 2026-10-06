"""Tests for mean-variance portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_advisor.optimizer import INSUFFICIENT_HISTORY_WARNING, optimize_portfolio, portfolio_stats


def test_thin_history_equal_weights_with_warning():
    thin = pd.DataFrame(
        np.random.default_rng(0).normal(0.001, 0.02, (5, 3)),
        columns=["A", "B", "C"],
    )
    weights, warning = optimize_portfolio(thin, "moderate")
    assert warning == INSUFFICIENT_HISTORY_WARNING
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(abs(w - 1 / 3) < 1e-6 for w in weights.values())


def test_slsqp_respects_max_weight_cap():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    cols = [f"T{i}" for i in range(8)]
    synthetic = pd.DataFrame(
        rng.normal(0.001, 0.02, (100, len(cols))),
        index=dates,
        columns=cols,
    )
    weights, warning = optimize_portfolio(synthetic, "conservative")
    assert warning is None
    assert abs(sum(weights.values()) - 1.0) < 1e-4
    assert all(w <= 0.15 + 1e-6 for w in weights.values())
    stats = portfolio_stats(weights, synthetic)
    assert "sharpe_ratio" in stats
