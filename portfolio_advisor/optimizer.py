"""Mean-variance portfolio optimization."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Historical sample means are noisy expected-return inputs.
# Black-Litterman (or similar shrinkage estimators) would be a future upgrade.

MIN_RETURN_ROWS = 30
INSUFFICIENT_HISTORY_WARNING = "Insufficient price history — using equal weights."

RISK_PARAMS: dict[str, dict[str, float]] = {
    "conservative": {"max_weight": 0.15, "risk_aversion": 5.0},
    "moderate": {"max_weight": 0.25, "risk_aversion": 2.0},
    "aggressive": {"max_weight": 0.35, "risk_aversion": 0.5},
}

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252


def _get_params(risk_tolerance: str) -> dict[str, float]:
    return RISK_PARAMS.get(risk_tolerance, RISK_PARAMS["moderate"])


def _equal_weights(tickers: list[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    weight = 1.0 / len(tickers)
    return {ticker: weight for ticker in tickers}


def optimize_portfolio(
    returns_df: pd.DataFrame,
    risk_tolerance: str,
) -> Tuple[Dict[str, float], str | None]:
    """Maximize expected return minus risk-aversion-weighted variance."""
    if returns_df.empty or returns_df.shape[1] == 0:
        return {}, None

    tickers = list(returns_df.columns)
    n = len(tickers)

    if len(returns_df) < MIN_RETURN_ROWS:
        return _equal_weights(tickers), INSUFFICIENT_HISTORY_WARNING

    if n == 1:
        return {tickers[0]: 1.0}, None

    params = _get_params(risk_tolerance)
    max_weight = params["max_weight"]
    risk_aversion = params["risk_aversion"]

    mu = returns_df.mean().values * TRADING_DAYS
    cov = returns_df.cov().values * TRADING_DAYS

    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = np.full(n, min(1.0 / n, max_weight))
    x0 = x0 / x0.sum()

    def objective(weights: np.ndarray) -> float:
        port_return = float(weights @ mu)
        port_variance = float(weights @ cov @ weights)
        return -(port_return - risk_aversion * port_variance)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    weights = result.x if result.success else x0
    if abs(weights.sum() - 1.0) > 1e-6 and weights.sum() > 0:
        weights = weights / weights.sum()

    return {ticker: float(weight) for ticker, weight in zip(tickers, weights)}, None


def portfolio_stats(weights: Dict[str, float], returns_df: pd.DataFrame) -> Dict[str, float]:
    """Compute annualized return, volatility, and Sharpe ratio."""
    if not weights or returns_df.empty:
        return {"expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers:
        return {"expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    subset = returns_df[tickers]

    mu = subset.mean().values * TRADING_DAYS
    cov = subset.cov().values * TRADING_DAYS

    expected_return = float(w @ mu)
    volatility = float(np.sqrt(w @ cov @ w))
    sharpe = (expected_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0.0

    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
    }


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")

    thin = pd.DataFrame(
        rng.normal(0.001, 0.02, (5, 3)),
        index=pd.date_range("2023-01-01", periods=5, freq="B"),
        columns=["A", "B", "C"],
    )
    thin_weights, thin_warning = optimize_portfolio(thin, "moderate")
    assert thin_warning == INSUFFICIENT_HISTORY_WARNING
    assert abs(sum(thin_weights.values()) - 1.0) < 1e-6
    assert all(abs(w - 1 / 3) < 1e-6 for w in thin_weights.values())

    for risk in ("conservative", "moderate", "aggressive"):
        n_assets = 8 if risk == "conservative" else 4
        cols = [f"T{i}" for i in range(n_assets)]
        synthetic = pd.DataFrame(
            rng.normal(0.001, 0.02, (100, n_assets)),
            index=dates,
            columns=cols,
        )
        weights, warning = optimize_portfolio(synthetic, risk)
        assert warning is None
        cap = _get_params(risk)["max_weight"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-4, f"Weights sum to {total}, expected ~1.0"
        assert all(w <= cap + 1e-6 for w in weights.values())
        stats = portfolio_stats(weights, synthetic)
        assert "sharpe_ratio" in stats

    print("optimizer checkpoint passed")
