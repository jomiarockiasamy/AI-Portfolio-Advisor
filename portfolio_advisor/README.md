# AI Portfolio Advisor (app)

The main project README lives at the repo root: [`../README.md`](../README.md) (live URL, screenshot, limits, deploy).

This folder contains the Streamlit app and Python modules.

## Run locally

From the repo root with venv activated:

```bash
streamlit run portfolio_advisor/app.py
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Default LLM backend |
| `ANTHROPIC_API_KEY` | Optional LLM backend |
| `LLM_PROVIDER` | Force `groq` or `anthropic` |
| `WEBULL_*` | Positions import only (local — not for public Streamlit secrets) |
| `FINNHUB_API_KEY` | Optional thesis enrichment |

Copy [`.env.example`](../.env.example) to `.env` at the repo root.

## Modules

| File | Role |
|------|------|
| `app.py` | Streamlit UI (Build + Analyze) |
| `screener.py` | Rule-based universe filter |
| `optimizer.py` | Mean–variance (SciPy SLSQP) |
| `signal_engine.py` | Composite BUY/HOLD/SELL + score breakdown |
| `rebalance_engine.py` | Signal-aligned tilt + weight caps |
| `holdings.py` | Review + rebalance table |
| `llm_analyst.py` | Groq/Anthropic theses |
| `risk_profile.py` | Rule-based portfolio risk label |
| `db.py` | SQLite portfolios, thesis cache, rate limit |
| `webull_client.py` | Optional Webull positions import |
| `fundamentals_finnhub.py` | Optional Finnhub data |
| `pages/2_Methodology.py` | In-app methodology |

See [`WEBULL_API_NOTES.md`](WEBULL_API_NOTES.md) for Webull API setup.

## Module checkpoints

```bash
python -m portfolio_advisor.optimizer
python -m portfolio_advisor.signal_engine
python -m portfolio_advisor.rebalance_engine
python -m portfolio_advisor.formatters
```
