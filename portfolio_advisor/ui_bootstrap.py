"""Shared Streamlit bootstrap for main app and multipage routes."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_advisor import db

DISCLAIMER = (
    "This tool is for research and educational purposes only. "
    "It is not licensed financial advice and does not execute trades."
)

_SECRET_KEYS = (
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_PROVIDER",
    "WEBULL_APP_KEY",
    "WEBULL_APP_SECRET",
    "FINNHUB_API_KEY",
    "WEBULL_ENV",
    "WEBULL_ACCOUNT_ID",
)


def load_dotenv() -> None:
    """Load project .env into os.environ when present (local dev)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_secrets_into_env() -> None:
    """Inject Streamlit Cloud secrets into os.environ for downstream modules."""
    try:
        for key in _SECRET_KEYS:
            if key not in os.environ and key in st.secrets:
                os.environ[key] = st.secrets[key]
    except Exception:
        return


def init_session_state() -> None:
    if "rate_limit_session_id" not in st.session_state:
        st.session_state.rate_limit_session_id = str(uuid.uuid4())


def bootstrap_app(*, page_title: str = "AI Portfolio Advisor", layout: str = "wide") -> None:
    st.set_page_config(page_title=page_title, layout=layout)
    load_dotenv()
    load_secrets_into_env()
    db.init_db()
    init_session_state()


def render_disclaimer() -> None:
    st.caption(DISCLAIMER)
