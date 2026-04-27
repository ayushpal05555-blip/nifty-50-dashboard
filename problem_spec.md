# Problem Specification

## Objective
Predict the probability that NIFTY 50 monthly forward return is negative.

## Target
- `y_1m_down = 1` if `forward_return_21d < 0`, else `0`
- `forward_return_21d = close(t+21)/close(t) - 1`

## Data Frequency
- Daily OHLCV for `^NSEI`
- Daily close for `^INDIAVIX`

## Prediction Schedule
- Daily EOD IST inference
- Weekly retraining (recommended cron)

## Primary ML Metrics
- ROC AUC
- Brier score
- Precision for downside class (`label=1`)

## Trading Metrics
- Cumulative return
- Sharpe (daily, annualized)
- Max drawdown

## Initial Signal Policy
- `P(down) >= 0.60` => `BEARISH`
- `0.45 <= P(down) < 0.60` => `NEUTRAL`
- `P(down) < 0.45` => `BULLISH`

## Validation Method
- Walk-forward expanding windows
- No random train/test split

