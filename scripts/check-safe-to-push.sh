#!/usr/bin/env bash
# Block commits that would publish secrets, tokens, logs, or local DBs.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

FAIL=0

block_path() {
  local pattern="$1"
  local desc="$2"
  if git diff --cached --name-only | grep -qE "$pattern"; then
    echo "BLOCKED: staged file matches $desc ($pattern)"
    git diff --cached --name-only | grep -E "$pattern" || true
    FAIL=1
  fi
}

block_path '^\.env$' 'real .env file'
block_path '^conf/' 'Webull token directory'
block_path '\.(log|db)$' 'log or SQLite database'
block_path '^\.streamlit/secrets\.toml$' 'Streamlit secrets'
block_path '^\.venv/' 'virtualenv'

# Scan staged text for common live key prefixes (not placeholders in docs)
STAGED_DIFF="$(git diff --cached -U0 --no-color -- . ':(exclude)scripts/check-safe-to-push.sh' 2>/dev/null || true)"
if echo "$STAGED_DIFF" | grep -qE 'gsk_[A-Za-z0-9]{10,}|sk-ant-api[0-9A-Za-z_-]{10,}|sk-proj-[A-Za-z0-9]{10,}'; then
  echo "BLOCKED: staged diff looks like a live API key (gsk_/sk-ant/sk-proj)."
  FAIL=1
fi

# Home-directory paths in staged adds often come from log paste mistakes
if echo "$STAGED_DIFF" | grep -qE '/Users/[A-Za-z0-9._-]+/'; then
  echo "BLOCKED: staged diff contains a macOS home-directory path (often from logs or local notes)."
  FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo ""
  echo "Fix staging, then retry. Safe files only: code, tests, .env.example (placeholders)."
  exit 1
fi

echo "OK: no secrets, logs, conf/, .env, or *.db staged."
