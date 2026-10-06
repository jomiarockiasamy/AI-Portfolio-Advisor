"""CLI utility to clear rate-limit counters in portfolios.db."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_advisor import db


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset portfolio advisor rate limits.")
    parser.add_argument(
        "--action",
        choices=["build", "analyze", "all"],
        default="all",
        help="Which action counter to clear (default: all).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Clear only rows whose identifier contains this session UUID.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to portfolios.db (default: portfolio_advisor/portfolios.db).",
    )
    args = parser.parse_args()

    if args.db:
        db.DB_PATH = Path(args.db)

    deleted = db.clear_rate_limit_log(action=args.action, session_id=args.session_id)
    print(f"Cleared {deleted} rate-limit row(s) for action={args.action}.")


if __name__ == "__main__":
    main()
