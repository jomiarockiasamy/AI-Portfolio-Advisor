# Webull OpenAPI — Step 0 Pre-flight Findings

Verified: **2026-07-26** against live docs at [developer.webull.com](https://developer.webull.com) and installed SDK **webull-openapi-python-sdk 2.0.15**.

## Product decision (shipped design)

**Webull = positions import only (Trading API). All pricing via yfinance.**

- No `DataClient`, snapshot, streaming, or paid OpenAPI Market Data subscription path in this app.
- No "live" price badge — prices are delayed/cached yfinance data, not Webull quotes.
- This is intentional product scope (keeps the app free), not a temporary fallback.

## SDK verification

| Item | Status |
|------|--------|
| Package | `pip install webull-openapi-python-sdk` → v2.0.15 |
| Trading auth | `ApiClient(app_key, app_secret, "us")` + `add_endpoint("us", host)` |
| Trade client | `TradeClient(api_client)` |
| Account list | `trade_client.account_v2.get_account_list()` |
| Positions | `trade_client.account_v2.get_account_position(account_id)` |
| Market data client | `DataClient(api_client)` — **not used in shipped code** |
| Snapshot (docs only) | `data_client.market_data.get_snapshot(symbols, category, ...)` |

Sandbox host: `api.sandbox.webull.com`  
Production host: `api.webull.com`

Official SDK verify snippet ([SDK docs](https://developer.webull.com/apis/docs/sdk.md)):

```python
from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

api_client = ApiClient("<app_key>", "<app_secret>", "us")
api_client.add_endpoint("us", "api.sandbox.webull.com")
trade_client = TradeClient(api_client)
res = trade_client.account_v2.get_account_list()
```

## Positions response field mapping

Live sandbox call **not executed** — `WEBULL_APP_KEY` / `WEBULL_APP_SECRET` not present in local `.env` at verification time.

Documented field names from official docs + SDK references (map flexibly at parse time):

| App field | Likely API fields |
|-----------|-------------------|
| `ticker` | `symbol`, or `ticker.symbol`, or `ticker.disSymbol` |
| `shares` | `qty`, `quantity`, or `position` (string/number) |
| `avg_cost` | `avg_cost`, `costPrice`, or `average_cost` |

Response may be a JSON list or an object with a `positions` array. Pagination: some API versions support `page_size` / `last_instrument_id` (v2 US uses account_id query on `/openapi/assets/positions`).

## Market data pricing verification

Official [Market Data Getting Started](https://developer.webull.com/apis/market-data-api/getting-started.md):

> Accessing market data (both historical and real-time) for US stocks and ETFs **requires an active OpenAPI market data subscription**. If you receive a **403** error … subscribe first.

Snapshot HTTP shape ([Data API](https://developer.webull.com/apis/market-data-api/data-api.md)):

```
GET /openapi/market-data/stock/snapshot?symbols=AAPL&category=US_STOCK
```

SDK equivalent: `data_client.market_data.get_snapshot(["AAPL"], Category.US_STOCK.name)`.

**Live quote test:** Attempted `DataClient` init with placeholder credentials → **401 UNAUTHORIZED** (invalid credentials). With valid trading credentials but **no market-data subscription**, official docs indicate **403 subscription error** is the expected outcome for snapshot/history calls.

| Outcome | Decision |
|---------|----------|
| 403 / subscription error (expected with valid trade keys, no MD sub) | Positions-only + yfinance pricing — **shipped** |
| 401 without credentials | Cannot confirm live 403; rely on official docs — same decision |
| 200 + price (unlikely without subscription) | Still use yfinance in v1; record here; no live badge |

## Step 0 gate checklist

- [x] Live docs re-fetched; SDK imports/methods confirmed against installed package 2.0.15
- [x] Sandbox positions call: documented field mapping; live call pending user sandbox credentials
- [x] Sandbox quote/snapshot: docs + 401 on invalid creds; subscription requirement confirmed in official docs
- [x] **Confirmed: positions-only + yfinance pricing is the shipped design**

## Environment variables

```
WEBULL_APP_KEY=
WEBULL_APP_SECRET=
WEBULL_ENV=sandbox          # or production
WEBULL_ACCOUNT_ID=          # optional if multiple accounts
```

Apply for Individual API keys: [developer.webull.com](https://developer.webull.com)
