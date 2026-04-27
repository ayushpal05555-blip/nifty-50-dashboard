# NIFTY 50 Live Dashboard (Local Server)

## Run Dashboard

From this folder:

```bash
node server.mjs
```

Then open:

- http://localhost:8787

## Run ML Pipeline

Install Python deps:

```bash
python3 -m pip install -r ml/requirements.txt
```

Train + generate latest forecast artifact:

```bash
python3 ml/train_pipeline.py
```

Or via npm script:

```bash
npm run ml:train
```

## What this does

- Serves the dashboard HTML over HTTP (not `file://`)
- Exposes `GET /api/live` for:
  - NIFTY 50 live
  - India VIX live
  - computed MTD return
- Exposes `GET /api/forecast` for:
  - latest model probabilities
  - ensemble signal
  - validation summary
  - backtest summary
- Exposes `GET /api/monitoring` for:
  - drift/calibration checks
  - live-accuracy availability checks
  - retrain trigger and rollback recommendation
- Exposes `GET /api/health` for:
  - app readiness and forecast freshness
  - retrain runtime status
  - monitoring summary status
- Dashboard polls `/api/live` every 5 minutes and updates ticker/footer/live strip

## Notes

- Data source: Yahoo Finance chart endpoint (`^NSEI`, `^INDIAVIX`)
- If live fetch fails, API returns fallback mode and dashboard keeps static values.
- Forecasts are read from `ml/artifacts/latest_forecast.json`.
- Monitoring policy/checklist is documented in `ml/monitoring_governance.md`.
