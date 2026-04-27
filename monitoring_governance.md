# Step 12: Monitoring + Governance

Use this checklist to operationalize model drift monitoring, live quality checks, retrain triggers, and rollback decisions.

## Monitoring checklist

- **Artifact freshness**: `latest_forecast.json` age must be <= `FORECAST_STALE_HOURS`.
- **Schema validity**: required forecast keys must exist (`generated_at`, `ensemble_down`, `validation`, `feature_snapshot`, `backtest`, etc.).
- **Data drift proxy**:
  - `feature_snapshot.vol_21d <= 0.08`
  - `abs(feature_snapshot.vix_norm_21d) <= 2.0`
- **Calibration drift**:
  - worst model `validation.*.brier_mean <= 0.25`
- **Model disagreement**:
  - spread across model downside probs (`max-min`) <= `0.12`
- **Signal stability**:
  - `abs(explainability.signal_change.delta_ensemble_down_pp) <= 8`
- **Live accuracy**:
  - once available, track `live_monitoring.hit_rate` (target >= `0.52`)
  - alert hard if `< 0.48` for 3 consecutive checks

## Alerts and severity

- **OK**: metric within guardrail
- **WARN**: guardrail breached but service still operational
- **CRITICAL**: missing/invalid artifact or essential metric unavailable

Status rollup:

- `CRITICAL`: at least one critical check
- `WARN`: no critical checks but 2 or more warn checks
- `OK`: otherwise

## Retrain trigger policy

Trigger retrain when:

- monitoring status is `CRITICAL`, or
- warn checks count is >= 2

Do not trigger a second retrain while one is already running.

## Rollback policy

Rollback to last known good model artifacts if any is true:

- post-retrain artifact is invalid or missing required fields
- post-retrain monitoring has more `CRITICAL` alerts than pre-retrain
- live hit-rate < 0.48 for 3 consecutive windows

Rollback target:

- previous `ml/artifacts/model_bundle.joblib`
- previous `ml/artifacts/latest_forecast.json`

## API outputs

- `GET /api/monitoring` returns:
  - check-level results
  - alert list
  - status, retrain trigger, rollback recommendation
  - governance policy block
- `GET /api/health` includes a monitoring summary for quick readiness checks.
