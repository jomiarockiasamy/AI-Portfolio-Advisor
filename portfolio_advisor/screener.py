"""Rule-based stock screening and scoring."""

from __future__ import annotations

import pandas as pd

RISK_BETA_CAPS: dict[str, float | None] = {
    "conservative": 1.0,
    "moderate": 1.4,
    "aggressive": None,
}

STYLE_WEIGHTS: dict[str, tuple[float, float]] = {
    "growth": (0.7, 0.3),
    "value": (0.3, 0.7),
    "blend": (0.5, 0.5),
}


def _rank_percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average").fillna(0.0)


def screen_candidates(
    fundamentals: pd.DataFrame,
    risk_tolerance: str,
    style: str,
    exclude_sectors: list[str],
    max_candidates: int,
) -> pd.DataFrame:
    """Filter and rank stocks by risk profile and style tilt."""
    if fundamentals.empty:
        return fundamentals.copy()

    df = fundamentals.copy()
    if "ticker" not in df.columns:
        df = df.reset_index().rename(columns={"index": "ticker"})

    if exclude_sectors:
        df = df[~df["sector"].isin(exclude_sectors)]

    beta_cap = RISK_BETA_CAPS.get(risk_tolerance)
    if beta_cap is not None:
        df = df[df["beta"].notna() & (df["beta"] <= beta_cap)]

    if risk_tolerance == "conservative":
        df = df[df["debtToEquity"].notna() & (df["debtToEquity"] <= 1.0)]

    if df.empty:
        return df

    growth_signal = _rank_percentile(df["revenueGrowth"].fillna(0)) + _rank_percentile(
        df["earningsGrowth"].fillna(0)
    )

    pe = df["trailingPE"].copy()
    inverse_pe = pe.where(pe > 0, other=pd.NA)
    value_signal = _rank_percentile(1.0 / inverse_pe)

    growth_weight, value_weight = STYLE_WEIGHTS.get(style, STYLE_WEIGHTS["blend"])
    df = df.copy()
    df["score"] = growth_weight * growth_signal + value_weight * value_signal

    df = df.sort_values("score", ascending=False)
    return df.head(max_candidates).reset_index(drop=True)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    n = 50
    sectors = ["Technology", "Healthcare", "Financial Services", "Energy", "Consumer"]
    synthetic = pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(n)],
            "sector": rng.choice(sectors, n),
            "beta": rng.uniform(0.5, 2.0, n),
            "debtToEquity": rng.uniform(0.0, 2.5, n),
            "trailingPE": rng.uniform(5, 40, n),
            "revenueGrowth": rng.uniform(-0.1, 0.5, n),
            "earningsGrowth": rng.uniform(-0.2, 0.6, n),
        }
    )

    exclude = ["Energy"]
    result = screen_candidates(
        synthetic,
        risk_tolerance="moderate",
        style="blend",
        exclude_sectors=exclude,
        max_candidates=10,
    )

    assert len(result) <= 10
    assert not (result["sector"] == "Energy").any()
    print("screener checkpoint passed")
