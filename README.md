# Helix-Consumption — staff review build

Locked-down deployment of IMC's consumption analysis model for Helix Water District's
named staff reviewers (Jennifer and Timothy) to sign off on rates before the Board sees
them. Built by the `cons-bill-calc-hosting` skill from IMC's `consumption-analysis`
project.

- Full parity with the internal tool for exploration: usage filtering by class, meter
  size, and tier; peaking; bill impact; impact distribution; and the full Affordability
  tab (heat map, tract picker, income-basis toggle).
- Rate and rate-structure editing does **not** exist in this build — removed from the
  code, not hidden by convention. Rate changes are Beeb's to make and redeploy.
- Pinned to Helix's own config and account cache (`AGENCY_SLUG` in `app.py`) — never a
  multi-agency picker, even if this repo is reused as a template later.
- Gated behind a shared passcode (`PASSCODE` env var, set on Render — never in source).

## Deploy

```
Build:  pip install -r requirements.txt
Start:  streamlit run src/consumption_analysis/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

Env var required: `PASSCODE`.

## Data

`build/` and `clients/helix/geo/` hold the account-level cache and Census join used at
runtime. `clients/helix/geo/accounts_address.csv` (raw street addresses) is deliberately
**not** included — nothing in the app reads it at runtime; only the offline geocoding
step in the source project needs it.
