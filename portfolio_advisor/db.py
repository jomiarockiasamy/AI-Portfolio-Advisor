"""SQLite persistence for portfolios, theses, and caching."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent / "portfolios.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create all tables if they do not exist."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                capital REAL NOT NULL,
                risk_tolerance TEXT NOT NULL,
                style TEXT NOT NULL,
                horizon TEXT NOT NULL,
                excluded_sectors TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS holdings (
                portfolio_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                weight REAL NOT NULL,
                dollar_amount REAL NOT NULL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            );

            CREATE TABLE IF NOT EXISTS theses (
                portfolio_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                moat TEXT,
                growth_drivers TEXT,
                key_risks TEXT,
                thesis_summary TEXT,
                conviction TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            );

            CREATE TABLE IF NOT EXISTS thesis_cache (
                ticker TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_holdings_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                holdings TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rate_limit_log (
                identifier TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rate_limit_identifier_ts
                ON rate_limit_log(identifier, timestamp);
            """
        )


def check_and_record_rate_limit(identifier: str, max_per_hour: int = 5) -> bool:
    """Return True if under cap and record this action; False if rate limited."""
    init_db()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=1)).isoformat()
    cleanup_cutoff = (now - timedelta(hours=24)).isoformat()

    with _connect() as conn:
        conn.execute(
            "DELETE FROM rate_limit_log WHERE timestamp < ?",
            (cleanup_cutoff,),
        )
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM rate_limit_log
            WHERE identifier = ? AND timestamp >= ?
            """,
            (identifier, cutoff),
        ).fetchone()
        count = int(row["cnt"]) if row else 0
        if count >= max_per_hour:
            conn.commit()
            return False

        conn.execute(
            "INSERT INTO rate_limit_log (identifier, timestamp) VALUES (?, ?)",
            (identifier, now.isoformat()),
        )
        conn.commit()
        return True


def save_portfolio(
    capital: float,
    risk_tolerance: str,
    style: str,
    horizon: str,
    excluded_sectors: list[str],
    weights: Dict[str, float],
    theses: List[Dict[str, Any]],
) -> int:
    """Persist a built portfolio with holdings and optional theses."""
    init_db()
    created_at = _utcnow_iso()

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO portfolios
                (created_at, capital, risk_tolerance, style, horizon, excluded_sectors)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                capital,
                risk_tolerance,
                style,
                horizon,
                json.dumps(excluded_sectors),
            ),
        )
        portfolio_id = int(cursor.lastrowid)

        for ticker, weight in weights.items():
            conn.execute(
                """
                INSERT INTO holdings (portfolio_id, ticker, weight, dollar_amount)
                VALUES (?, ?, ?, ?)
                """,
                (portfolio_id, ticker, weight, capital * weight),
            )

        for thesis in theses:
            if thesis.get("error"):
                continue
            conn.execute(
                """
                INSERT INTO theses
                    (portfolio_id, ticker, moat, growth_drivers, key_risks,
                     thesis_summary, conviction, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio_id,
                    thesis.get("ticker", ""),
                    thesis.get("moat", ""),
                    thesis.get("growth_drivers", ""),
                    thesis.get("key_risks", ""),
                    thesis.get("thesis_summary", ""),
                    thesis.get("conviction", ""),
                    created_at,
                ),
            )

        conn.commit()
        return portfolio_id


def list_portfolios() -> List[Dict[str, Any]]:
    """Return summary rows for all saved portfolios."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, capital, risk_tolerance, style, horizon, excluded_sectors
            FROM portfolios
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_portfolio_detail(portfolio_id: int) -> Dict[str, Any]:
    """Return portfolio metadata plus holdings and theses."""
    init_db()
    with _connect() as conn:
        portfolio = conn.execute(
            "SELECT * FROM portfolios WHERE id = ?",
            (portfolio_id,),
        ).fetchone()
        if portfolio is None:
            return {}

        holdings = conn.execute(
            "SELECT ticker, weight, dollar_amount FROM holdings WHERE portfolio_id = ?",
            (portfolio_id,),
        ).fetchall()
        theses = conn.execute(
            """
            SELECT ticker, moat, growth_drivers, key_risks, thesis_summary, conviction, created_at
            FROM theses WHERE portfolio_id = ?
            """,
            (portfolio_id,),
        ).fetchall()

    detail = dict(portfolio)
    detail["excluded_sectors"] = json.loads(detail.get("excluded_sectors") or "[]")
    detail["holdings"] = [dict(row) for row in holdings]
    detail["theses"] = [dict(row) for row in theses]
    return detail


def get_cached_thesis(ticker: str, ttl_days: int = 7) -> Optional[Dict[str, Any]]:
    """Return cached thesis if present and within TTL, else None."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT data, updated_at FROM thesis_cache WHERE ticker = ?",
            (ticker.upper(),),
        ).fetchone()

    if row is None:
        return None

    updated_at = datetime.fromisoformat(row["updated_at"])
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) - updated_at > timedelta(days=ttl_days):
        return None

    return json.loads(row["data"])


def save_thesis_cache(ticker: str, thesis: Dict[str, Any]) -> None:
    """Upsert a thesis into the cache."""
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO thesis_cache (ticker, data, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                data = excluded.data,
                updated_at = excluded.updated_at
            """,
            (ticker.upper(), json.dumps(thesis), _utcnow_iso()),
        )
        conn.commit()


def save_holdings_snapshot(holdings: List[Dict[str, Any]]) -> int:
    """Persist a user-entered holdings snapshot for future session use."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO user_holdings_snapshots (created_at, holdings)
            VALUES (?, ?)
            """,
            (_utcnow_iso(), json.dumps(holdings)),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_holdings_snapshots() -> List[Dict[str, Any]]:
    """List saved holdings snapshots."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, holdings
            FROM user_holdings_snapshots
            ORDER BY created_at DESC
            """
        ).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["holdings"] = json.loads(item["holdings"])
        results.append(item)
    return results


if __name__ == "__main__":
    import os
    import tempfile

    test_db = Path(tempfile.mkdtemp()) / "test.db"
    original = DB_PATH
    globals()["DB_PATH"] = test_db

    try:
        init_db()
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        theses = [
            {
                "ticker": "AAPL",
                "moat": "Brand",
                "growth_drivers": "Services",
                "key_risks": "Regulation",
                "thesis_summary": "Quality compounder",
                "conviction": "high",
            }
        ]
        pid = save_portfolio(
            capital=10000,
            risk_tolerance="moderate",
            style="blend",
            horizon="5y",
            excluded_sectors=["Energy"],
            weights=weights,
            theses=theses,
        )
        detail = get_portfolio_detail(pid)
        assert len(detail["holdings"]) == 2
        assert detail["holdings"][0]["ticker"] in weights
        assert len(detail["theses"]) == 1

        thesis = {
            "ticker": "AAPL",
            "moat": "Ecosystem",
            "growth_drivers": "AI",
            "key_risks": "Competition",
            "thesis_summary": "Test",
            "conviction": "medium",
        }
        save_thesis_cache("AAPL", thesis)
        cached = get_cached_thesis("AAPL")
        assert cached is not None
        assert cached["moat"] == "Ecosystem"

        with _connect() as conn:
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            conn.execute(
                "UPDATE thesis_cache SET updated_at = ? WHERE ticker = ?",
                (old_time, "AAPL"),
            )
            conn.commit()
        assert get_cached_thesis("AAPL") is None

        ident = "test-user-123"
        results = [check_and_record_rate_limit(ident, max_per_hour=5) for _ in range(6)]
        assert results[:5] == [True] * 5
        assert results[5] is False

        print("db checkpoint passed")
    finally:
        globals()["DB_PATH"] = original
