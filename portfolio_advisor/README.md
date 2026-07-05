# AI Portfolio Advisor

A Python + Streamlit app that screens stocks, optimizes portfolio allocations, and optionally generates AI investment theses. Built as a free public research tool with cost controls baked in.

## Features

- **Build New Portfolio** — screen ~30 large-cap stocks, mean-variance optimize weights, save to SQLite
- **Analyze My Portfolio** — review current holdings, sector concentration, and rebalance suggestions
- **AI Theses** — Groq (default, free tier) or Anthropic (optional override)
- **Thesis cache** — SQLite cache for 7 days; repeat ticker lookups skip the LLM
- **Rate limiting** — 5 build/analyze actions per hour, persisted in SQLite (IP-based on Cloud)

## Setup

```bash
cd /path/to/Stocks
python3 -m venv .venv
source .venv/bin/activate
pip install -r portfolio_advisor/requirements.txt
cp .env.example .env   # then edit with your keys
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Default LLM backend (recommended — free tier) |
| `ANTHROPIC_API_KEY` | Optional fallback LLM backend |
| `LLM_PROVIDER` | Force `groq` or `anthropic` |

Provider resolution order:
1. `LLM_PROVIDER` if set to `groq` or `anthropic`
2. Groq if `GROQ_API_KEY` is set
3. Anthropic if `ANTHROPIC_API_KEY` is set
4. No provider (thesis generation disabled with a warning)

```bash
export GROQ_API_KEY=your_key_here
```

## Run locally

```bash
streamlit run portfolio_advisor/app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (see root `.gitignore` — never commit `.env` or `*.db`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point at your repo with main file: `portfolio_advisor/app.py`.
4. In the app's **Secrets** settings (not in code), add:

```toml
GROQ_API_KEY = "your_actual_key"
```

Optionally:

```toml
ANTHROPIC_API_KEY = "your_anthropic_key"
LLM_PROVIDER = "groq"
```

The app loads secrets via `st.secrets` into `os.environ` at startup.

## Cost control

- **Groq default** — uses the free-tier Llama 3.3 70B model to minimize API spend on a public tool
- **7-day thesis cache** — cached theses are stored in SQLite (`thesis_cache` table); identical tickers within 7 days do not re-call the LLM
- **SQLite-backed rate limit** — max 5 build/analyze actions per rolling hour per client IP (`X-Forwarded-For` on Streamlit Cloud) or session UUID locally; survives page refresh within a running instance

## Ephemeral storage on Streamlit Cloud

Streamlit Community Cloud uses an **ephemeral filesystem**. The SQLite file (`portfolios.db`) is reset whenever the app cold-starts or redeploys. This means:

- Saved portfolios disappear after redeploy
- Thesis cache is cleared after redeploy
- Rate-limit history resets after redeploy

This is acceptable for a v1 free demo. For durable persistence across restarts, a future upgrade would move to a hosted database (e.g. Turso or Supabase free tier).

## Project structure

```
portfolio_advisor/
├── data_fetcher.py   # yfinance price + fundamentals (with retry)
├── screener.py       # rule-based filtering + scoring
├── llm_analyst.py    # Groq/Anthropic thesis generation
├── optimizer.py      # mean-variance optimization (scipy SLSQP)
├── holdings.py       # holdings review + rebalance actions
├── db.py             # SQLite persistence + thesis cache + rate limit
├── app.py            # Streamlit UI (two modes)
└── requirements.txt
```

## Limitations

- Mean-variance optimization uses historical mean returns as expected-return inputs — noisy in practice. Black-Litterman is noted as a future upgrade in code comments.
- Thin price history (< 30 days of returns) falls back to equal weighting with a visible warning.
- Invalid ticker symbols in Mode 2 are skipped with a warning; valid holdings still render.
- This is a research/allocation tool only — no brokerage or trade execution.
- Not licensed financial advice — a disclaimer is shown in the UI footer.

## Module checkpoints

Each module includes a `if __name__ == "__main__"` block for standalone verification:

```bash
python -m portfolio_advisor.data_fetcher
python -m portfolio_advisor.screener
python -m portfolio_advisor.optimizer
python -m portfolio_advisor.db
python -m portfolio_advisor.llm_analyst
python -m portfolio_advisor.holdings
```
