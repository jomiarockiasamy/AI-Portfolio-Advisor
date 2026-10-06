# Contributing

## Privacy before you push

Never commit `.env`, `conf/` (Webull tokens), `*.log`, or `*.db`. They are in `.gitignore`.

Before `git push`, run:

```bash
./scripts/check-safe-to-push.sh   # after git add, checks the index
git config core.hooksPath .githooks   # optional: auto-run the same check on every commit
```

Public Streamlit deploy: use **Groq/Finnhub** secrets only — not Webull keys.
