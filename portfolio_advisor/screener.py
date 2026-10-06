"""Rule-based stock screening and scoring."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

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


def _diagnose_filters(df: pd.DataFrame, risk_tolerance: str) -> dict[str, int]:
    """Return row counts after each filter stage (for debugging)."""
    counts: dict[str, int] = {"initial": len(df)}
    working = df.copy()

    beta_cap = RISK_BETA_CAPS.get(risk_tolerance)
    if beta_cap is not None:
        working = working[working["beta"].notna() & (working["beta"] <= beta_cap)]
    counts["after_beta"] = len(working)

    if risk_tolerance == "conservative":
        de = working["debtToEquity"].dropna()
        if len(de) >= 3:
            cutoff = de.quantile(0.50)
            working = working[
                working["debtToEquity"].notna() & (working["debtToEquity"] <= cutoff)
            ]
        counts["after_debt"] = len(working)
    else:
        counts["after_debt"] = len(working)

    return counts


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
        # Relative ranking within today's universe — NOT an absolute low-debt safety bar.
        # Keeps the lower-D/E half so conservative mode doesn't return zero candidates
        # when yfinance reports D/E as a percentage-scale number (e.g. 30–200).
        de = df["debtToEquity"].dropna()
        if len(de) >= 3:
            cutoff = de.quantile(0.50)
            df = df[df["debtToEquity"].notna() & (df["debtToEquity"] <= cutoff)]

    if df.empty:
        diagnosis = _diagnose_filters(fundamentals, risk_tolerance)
        logger.debug("screen_candidates empty after filters: %s", diagnosis)
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


def find_sector_alternatives(
    selected_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    risk_tolerance: str,
    style: str,
    exclude_sectors: list[str],
    max_alternates: int = 2,
) -> dict[str, list[dict]]:
    """For each selected ticker, return next-best same-sector alternatives not in the pick list."""
    if selected_df.empty or universe_df.empty:
        return {}

    selected_tickers = set(selected_df["ticker"].astype(str).str.upper())
    alternatives: dict[str, list[dict]] = {}

    for _, row in selected_df.iterrows():
        ticker = str(row["ticker"]).upper()
        sector = row.get("sector")
        if not sector or pd.isna(sector):
            continue

        sector_pool = universe_df[
            (universe_df["sector"] == sector)
            & (~universe_df["ticker"].astype(str).str.upper().isin(selected_tickers))
        ].copy()
        if sector_pool.empty:
            continue

        ranked = screen_candidates(
            sector_pool,
            risk_tolerance=risk_tolerance,
            style=style,
            exclude_sectors=exclude_sectors,
            max_candidates=len(sector_pool),
        )
        alt_rows: list[dict] = []
        for _, alt in ranked.head(max_alternates).iterrows():
            alt_rows.append(
                {
                    "ticker": alt["ticker"],
                    "shortName": alt.get("shortName"),
                    "trailingPE": alt.get("trailingPE"),
                    "forwardPE": alt.get("forwardPE"),
                    "revenueGrowth": alt.get("revenueGrowth"),
                    "earningsGrowth": alt.get("earningsGrowth"),
                    "beta": alt.get("beta"),
                    "score": alt.get("score"),
                }
            )
        if alt_rows:
            alternatives[ticker] = alt_rows

    return alternatives


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
            "debtToEquity": rng.uniform(20.0, 150.0, n),
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

    conservative = screen_candidates(
        synthetic,
        risk_tolerance="conservative",
        style="blend",
        exclude_sectors=[],
        max_candidates=10,
    )
    assert len(conservative) >= 3

    from portfolio_advisor.data_fetcher import DEFAULT_UNIVERSE, fetch_fundamentals

    live = fetch_fundamentals(DEFAULT_UNIVERSE)
    diagnosis = _diagnose_filters(live, "conservative")
    print("conservative filter diagnosis:", diagnosis)
    live_conservative = screen_candidates(
        live, risk_tolerance="conservative", style="growth", exclude_sectors=[], max_candidates=8
    )
    assert len(live_conservative) >= 3, f"Expected >=3 conservative candidates, got {len(live_conservative)}"
    print("screener checkpoint passed")
