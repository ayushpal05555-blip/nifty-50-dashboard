from __future__ import annotations

import json
import math
import pathlib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, roc_auc_score


ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_yahoo_chart(symbol: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    base = "https://query1.finance.yahoo.com/v8/finance/chart/"
    query = urllib.parse.urlencode({"range": period, "interval": interval, "events": "div,splits"})
    url = f"{base}{urllib.parse.quote(symbol)}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    ).dropna(subset=["close"])
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_dataset() -> pd.DataFrame:
    nifty = fetch_yahoo_chart("^NSEI", "10y", "1d")
    vix = fetch_yahoo_chart("^INDIAVIX", "10y", "1d")[["date", "close"]].rename(columns={"close": "vix_close"})
    df = nifty.merge(vix, on="date", how="left").sort_values("date").reset_index(drop=True)
    df["vix_close"] = df["vix_close"].ffill()

    df["ret_1d"] = df["close"].pct_change()
    df["ret_2d"] = df["close"].pct_change(2)
    df["ret_5d"] = df["close"].pct_change(5)
    df["mom_21d"] = df["close"] / df["close"].shift(21) - 1
    df["mom_126d"] = df["close"] / df["close"].shift(126) - 1
    df["ma_63"] = df["close"].rolling(63).mean()
    df["ma_126"] = df["close"].rolling(126).mean()
    df["dev_ma_63"] = df["close"] / df["ma_63"] - 1
    df["dev_ma_126"] = df["close"] / df["ma_126"] - 1
    df["vol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(21)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["vol_chg_21d"] = df["volume"].pct_change(21).replace([np.inf, -np.inf], np.nan)
    df["rsi_126"] = compute_rsi(df["close"], 14).rolling(126).mean()
    df["vix_norm_21d"] = df["vix_close"] / df["vix_close"].rolling(21).mean() - 1

    df["fwd_ret_21d"] = df["close"].shift(-21) / df["close"] - 1
    df["target_down_21d"] = (df["fwd_ret_21d"] < 0).astype(int)
    return df


FEATURES = [
    "ret_1d",
    "ret_2d",
    "ret_5d",
    "mom_21d",
    "mom_126d",
    "dev_ma_63",
    "dev_ma_126",
    "vol_21d",
    "hl_spread",
    "vol_chg_21d",
    "rsi_126",
    "vix_norm_21d",
]


@dataclass
class FoldResult:
    auc: float
    brier: float
    precision_down: float


@dataclass
class CalibrationFoldResult:
    auc_uncal: float
    brier_uncal: float
    auc_sigmoid: float
    brier_sigmoid: float
    auc_isotonic: float
    brier_isotonic: float


def walk_forward_indices(n_samples: int, min_train: int = 750, step: int = 63):
    start = min_train
    while start + step < n_samples:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, min(start + step, n_samples))
        yield train_idx, test_idx
        start += step


def get_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=8, random_state=42, class_weight="balanced_subsample"
        ),
        "gb": GradientBoostingClassifier(random_state=42),
    }


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")


def _fit_calibrated(base_model, method: str):
    # Uses internal CV on training window -> avoids using test window for calibration.
    return CalibratedClassifierCV(base_model, method=method, cv=3)


def evaluate(df: pd.DataFrame):
    data = df.dropna(subset=FEATURES + ["target_down_21d"]).reset_index(drop=True)
    X = data[FEATURES].values
    y = data["target_down_21d"].values.astype(int)
    models = get_models()
    fold_stats: dict[str, list[FoldResult]] = {k: [] for k in models}
    calib_stats: dict[str, list[CalibrationFoldResult]] = {k: [] for k in models}

    for train_idx, test_idx in walk_forward_indices(len(data)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for name, model in get_models().items():
            # Uncalibrated
            uncal = get_models()[name]
            uncal.fit(X_train, y_train)
            p_uncal = uncal.predict_proba(X_test)[:, 1]

            # Platt scaling (sigmoid)
            sig = _fit_calibrated(get_models()[name], method="sigmoid")
            sig.fit(X_train, y_train)
            p_sig = sig.predict_proba(X_test)[:, 1]

            # Isotonic (more flexible; needs more data, can overfit)
            iso = _fit_calibrated(get_models()[name], method="isotonic")
            iso.fit(X_train, y_train)
            p_iso = iso.predict_proba(X_test)[:, 1]

            # Keep legacy fold_stats as "sigmoid" (backwards compatible)
            pred = (p_sig >= 0.5).astype(int)
            fold_stats[name].append(
                FoldResult(
                    auc=_safe_auc(y_test, p_sig),
                    brier=brier_score_loss(y_test, p_sig),
                    precision_down=precision_score(y_test, pred, zero_division=0),
                )
            )

            calib_stats[name].append(
                CalibrationFoldResult(
                    auc_uncal=_safe_auc(y_test, p_uncal),
                    brier_uncal=brier_score_loss(y_test, p_uncal),
                    auc_sigmoid=_safe_auc(y_test, p_sig),
                    brier_sigmoid=brier_score_loss(y_test, p_sig),
                    auc_isotonic=_safe_auc(y_test, p_iso),
                    brier_isotonic=brier_score_loss(y_test, p_iso),
                )
            )
    return data, fold_stats, calib_stats


def assign_regimes(data: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    Simple regime definition:
    - Vol bucket: HIGHVOL if vol_21d >= 75th percentile
    - Trend bucket: UPTREND if dev_ma_126 >= 0 else DOWNTREND
    """
    vol = data["vol_21d"].astype(float)
    vol_thresh = float(vol.dropna().quantile(0.75)) if vol.notna().any() else 0.0
    vol_bucket = np.where(vol >= vol_thresh, "HIGHVOL", "LOWVOL")
    trend_bucket = np.where(data["dev_ma_126"].astype(float) >= 0, "UPTREND", "DOWNTREND")
    regime = pd.Series([f"{v}_{t}" for v, t in zip(vol_bucket, trend_bucket)], index=data.index)
    return regime, vol_thresh


def _choose_calibration_methods(calib_stats: dict[str, list[CalibrationFoldResult]]):
    chosen: dict[str, str] = {}
    summary: dict[str, dict] = {}
    for name, rows in calib_stats.items():
        b_uncal = np.mean([r.brier_uncal for r in rows]) if rows else float("nan")
        b_sig = np.mean([r.brier_sigmoid for r in rows]) if rows else float("nan")
        b_iso = np.mean([r.brier_isotonic for r in rows]) if rows else float("nan")
        best = min([("uncal", b_uncal), ("sigmoid", b_sig), ("isotonic", b_iso)], key=lambda x: x[1])[0]
        # Disallow isotonic if it barely appears better than sigmoid (stability heuristic)
        if best == "isotonic" and (b_sig - b_iso) < 0.001:
            best = "sigmoid"
        chosen[name] = best
        summary[name] = {
            "brier_mean_uncal": float(b_uncal),
            "brier_mean_sigmoid": float(b_sig),
            "brier_mean_isotonic": float(b_iso),
            "chosen": best,
            "brier_improvement_vs_uncal": float(b_uncal - min(b_sig, b_iso)),
        }
    return chosen, summary


def train_final_models(data: pd.DataFrame, chosen_methods: dict[str, str]):
    X = data[FEATURES].values
    y = data["target_down_21d"].values.astype(int)
    trained = {}
    for name, model in get_models().items():
        method = chosen_methods.get(name, "sigmoid")
        if method == "uncal":
            model.fit(X, y)
            trained[name] = model
        else:
            calibrated = CalibratedClassifierCV(model, method=method, cv=5)
            calibrated.fit(X, y)
            trained[name] = calibrated
    return trained


def _fit_model_with_method(model_name: str, method: str, X_train: np.ndarray, y_train: np.ndarray):
    base = get_models()[model_name]
    if method == "uncal":
        base.fit(X_train, y_train)
        return base
    calibrated = CalibratedClassifierCV(base, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def build_oos_predictions(data: pd.DataFrame, chosen_methods: dict[str, str], regimes: pd.Series):
    """
    Build walk-forward out-of-sample predictions per model using chosen calibration.
    Returns a frame with columns:
      date, y, regime, p_down_{model}
    """
    X = data[FEATURES].values
    y = data["target_down_21d"].values.astype(int)

    rows = []
    for train_idx, test_idx in walk_forward_indices(len(data)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        r_test = regimes.iloc[test_idx].values
        dates = pd.to_datetime(data.iloc[test_idx]["date"]).astype(str).values

        preds = {}
        for name in get_models().keys():
            method = chosen_methods.get(name, "sigmoid")
            mdl = _fit_model_with_method(name, method, X_train, y_train)
            preds[name] = mdl.predict_proba(X_test)[:, 1]

        for i in range(len(test_idx)):
            rows.append(
                {
                    "date": dates[i],
                    "y": int(y_test[i]),
                    "regime": str(r_test[i]),
                    **{f"p_down_{name}": float(preds[name][i]) for name in preds},
                }
            )
    return pd.DataFrame(rows)


def _weights_from_briers(briers: dict[str, float]) -> dict[str, float]:
    vals = {k: float(v) for k, v in briers.items() if np.isfinite(v) and v > 0}
    if not vals:
        n = len(get_models())
        return {k: 1.0 / n for k in get_models().keys()}
    inv = {k: 1.0 / v for k, v in vals.items()}
    s = sum(inv.values())
    return {k: inv.get(k, 0.0) / s for k in get_models().keys()}


def compute_ensemble_weights(oos: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Weights are inverse-Brier normalized.
    Returns:
      global_weights, regime_weights
    """
    model_names = list(get_models().keys())
    global_briers = {
        m: brier_score_loss(oos["y"].values, oos[f"p_down_{m}"].values) for m in model_names if len(oos) > 0
    }
    global_w = _weights_from_briers(global_briers)

    regime_w: dict[str, dict[str, float]] = {}
    for reg, grp in oos.groupby("regime"):
        b = {m: brier_score_loss(grp["y"].values, grp[f"p_down_{m}"].values) for m in model_names if len(grp) > 5}
        regime_w[str(reg)] = _weights_from_briers(b)
    return global_w, regime_w


def ensemble_predict_proba(
    model_probs: dict[str, float], current_regime: str, global_weights: dict[str, float], regime_weights: dict[str, dict[str, float]]
) -> tuple[float, dict[str, float]]:
    weights_used = regime_weights.get(current_regime, global_weights)
    ensemble_down = float(sum(weights_used.get(name, 0.0) * model_probs[name] for name in model_probs.keys()))
    return ensemble_down, weights_used


def _get_underlying_estimator(model):
    """
    For calibrated models, use first fitted base estimator for explainability proxy.
    """
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        cc = model.calibrated_classifiers_[0]
        if hasattr(cc, "estimator"):
            return cc.estimator
    return model


def _normalize_importance(vals: dict[str, float]) -> dict[str, float]:
    s = float(sum(max(v, 0.0) for v in vals.values()))
    if s <= 0:
        n = len(vals) if vals else 1
        return {k: 1.0 / n for k in vals.keys()}
    return {k: float(max(v, 0.0) / s) for k, v in vals.items()}


def compute_global_importance(models: dict, data: pd.DataFrame) -> dict:
    """
    SHAP-first global importance:
    - TreeExplainer for RF/GB
    - LinearExplainer for LogisticRegression
    Falls back to model-native importances/coefficients when SHAP is unavailable.
    """
    X = data[FEATURES].dropna()
    if len(X) > 500:
        X = X.tail(500)
    xvals = X.values
    per_model = {}
    agg = {f: 0.0 for f in FEATURES}
    method_used = "fallback"

    for name, mdl in models.items():
        est = _get_underlying_estimator(mdl)
        imp = {f: 0.0 for f in FEATURES}
        if shap is not None:
            try:
                if name in {"rf", "gb"}:
                    explainer = shap.TreeExplainer(est)
                    sv = explainer.shap_values(xvals)
                    arr = sv[1] if isinstance(sv, list) else sv
                    vals = np.mean(np.abs(arr), axis=0)
                elif name == "logreg":
                    explainer = shap.LinearExplainer(est, xvals)
                    sv = explainer.shap_values(xvals)
                    vals = np.mean(np.abs(sv), axis=0)
                else:
                    vals = np.zeros(len(FEATURES))
                imp = {f: float(v) for f, v in zip(FEATURES, vals)}
                method_used = "shap"
            except Exception:
                pass

        if method_used != "shap" or all(v == 0.0 for v in imp.values()):
            if hasattr(est, "feature_importances_"):
                vals = est.feature_importances_
                imp = {f: float(v) for f, v in zip(FEATURES, vals)}
            elif hasattr(est, "coef_"):
                coef = np.abs(np.ravel(est.coef_))
                imp = {f: float(v) for f, v in zip(FEATURES, coef)}
            else:
                imp = {f: 1.0 for f in FEATURES}

        imp = _normalize_importance(imp)
        per_model[name] = imp
        for f in FEATURES:
            agg[f] += imp.get(f, 0.0) / max(len(models), 1)

    agg = _normalize_importance(agg)
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "method": method_used,
        "global_feature_importance": {k: float(v) for k, v in top},
        "top_features": [{"feature": k, "importance": float(v)} for k, v in top[:8]],
        "per_model_feature_importance": per_model,
    }


def compute_signal_change_diagnostics(
    latest_feature_snapshot: dict[str, float],
    ensemble_down: float,
    importance: dict,
    previous_report: dict | None,
) -> dict:
    if not previous_report:
        return {
            "has_previous": False,
            "delta_ensemble_down_pp": None,
            "why_signal_changed": [],
            "summary": "No previous forecast artifact available for change diagnostics.",
        }

    prev_ens = float(previous_report.get("ensemble_down", np.nan))
    prev_feat = previous_report.get("feature_snapshot", {}) or {}
    if not np.isfinite(prev_ens):
        prev_ens = ensemble_down

    imp = importance.get("global_feature_importance", {})
    rows = []
    for f in FEATURES:
        cur = float(latest_feature_snapshot.get(f, np.nan))
        prv = float(prev_feat.get(f, np.nan)) if f in prev_feat else np.nan
        if not np.isfinite(cur) or not np.isfinite(prv):
            continue
        delta = cur - prv
        score = abs(delta) * float(imp.get(f, 0.0))
        rows.append({"feature": f, "delta": float(delta), "weighted_delta": float(score)})
    rows = sorted(rows, key=lambda r: r["weighted_delta"], reverse=True)[:6]

    delta_pp = float((ensemble_down - prev_ens) * 100.0)
    direction = "increased downside risk" if delta_pp > 0 else ("reduced downside risk" if delta_pp < 0 else "was unchanged")
    summary = f"Ensemble downside probability {direction} by {delta_pp:+.2f}pp vs previous run."
    return {
        "has_previous": True,
        "delta_ensemble_down_pp": delta_pp,
        "why_signal_changed": rows,
        "summary": summary,
    }


def backtest(
    data: pd.DataFrame,
    models: dict,
    chosen_methods: dict[str, str],
    global_weights: dict[str, float],
    regime_weights: dict[str, dict[str, float]],
    regimes: pd.Series,
):
    """
    Realistic-ish daily backtest assumptions:
    - Signal generated at close(t) from available features
    - Position entered for next session return (t+1)
    - Three-state positioning: short / flat / long
    - Costs applied on position change:
        commission_bps = 2 bps per side equivalent on turnover
        slippage_bps   = 5 bps per unit turnover
      total per-turnover cost = 7 bps * |delta_position|
    """
    X = data[FEATURES].values
    probs = {name: mdl.predict_proba(X)[:, 1] for name, mdl in models.items()}

    ensemble_down = []
    weights_history = []
    for i in range(len(data)):
        mp = {name: float(probs[name][i]) for name in probs}
        reg = str(regimes.iloc[i])
        p_down, w = ensemble_predict_proba(mp, reg, global_weights, regime_weights)
        ensemble_down.append(p_down)
        weights_history.append(w)
    ensemble_down = np.array(ensemble_down)

    raw_signal = np.where(ensemble_down >= 0.60, -1, np.where(ensemble_down < 0.45, 1, 0))
    # Execute on next bar
    position = pd.Series(raw_signal, index=data.index).shift(1).fillna(0.0).values
    asset_ret = data["ret_1d"].fillna(0.0).values

    turnover = np.abs(np.diff(position, prepend=0.0))
    total_cost_bps = 7.0
    cost_rate = (total_cost_bps / 10000.0) * turnover
    strat_ret = position * asset_ret - cost_rate

    equity = np.cumprod(1 + np.nan_to_num(strat_ret))
    running_max = np.maximum.accumulate(equity)
    dd = (equity / running_max) - 1
    sharpe = (np.mean(strat_ret) / (np.std(strat_ret) + 1e-9)) * np.sqrt(252)
    downside = strat_ret[strat_ret < 0]
    sortino = (np.mean(strat_ret) / (np.std(downside) + 1e-9)) * np.sqrt(252) if len(downside) else 0.0
    years = max(len(strat_ret) / 252.0, 1e-9)
    cagr = float(equity[-1] ** (1 / years) - 1) if len(equity) else 0.0

    # Keep artifact small: sample equity/drawdown curve points
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(data["date"]).astype(str),
            "equity": equity,
            "drawdown": dd,
            "position": position,
            "ensemble_down": ensemble_down,
        }
    )
    sample_n = 250
    if len(curve) > sample_n:
        idx = np.linspace(0, len(curve) - 1, sample_n).astype(int)
        curve = curve.iloc[idx]

    return {
        "assumptions": {
            "execution": "signal at close, execute next session",
            "position_rule": {"bearish": -1, "neutral": 0, "bullish": 1},
            "thresholds": {"short_if_pdown_gte": 0.60, "long_if_pdown_lt": 0.45},
            "commission_bps_per_turnover": 2.0,
            "slippage_bps_per_turnover": 5.0,
            "total_cost_bps_per_turnover": total_cost_bps,
        },
        "total_return_pct": float((equity[-1] - 1) * 100) if len(equity) else 0.0,
        "max_drawdown_pct": float(np.min(dd) * 100) if len(dd) else 0.0,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "cagr_pct": float(cagr * 100),
        "avg_daily_turnover": float(np.mean(turnover)) if len(turnover) else 0.0,
        "trades": int(np.sum(turnover > 0)),
        "win_rate_pct": float(np.mean(strat_ret > 0) * 100) if len(strat_ret) else 0.0,
        "equity_curve_sample": curve.to_dict(orient="records"),
    }


def summarize_fold_stats(fold_stats: dict[str, list[FoldResult]]):
    out = {}
    for name, stats in fold_stats.items():
        aucs = [s.auc for s in stats if not math.isnan(s.auc)]
        out[name] = {
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "brier_mean": float(np.mean([s.brier for s in stats])) if stats else None,
            "precision_down_mean": float(np.mean([s.precision_down for s in stats])) if stats else None,
            "folds": len(stats),
        }
    return out


def build_reliability_curve(y_true: np.ndarray, p_down: np.ndarray, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, p_down, n_bins=n_bins, strategy="uniform")
    return {
        "bins": int(n_bins),
        "mean_pred": [float(x) for x in mean_pred],
        "frac_pos": [float(x) for x in frac_pos],
    }


def main():
    prev_report = None
    prev_path = ARTIFACT_DIR / "latest_forecast.json"
    if prev_path.exists():
        try:
            prev_report = json.loads(prev_path.read_text(encoding="utf-8"))
        except Exception:
            prev_report = None

    raw = build_dataset()
    data, fold_stats, calib_stats = evaluate(raw)
    chosen_methods, calib_summary = _choose_calibration_methods(calib_stats)
    regimes, vol_thresh = assign_regimes(data)
    oos = build_oos_predictions(data, chosen_methods, regimes)
    global_w, regime_w = compute_ensemble_weights(oos)

    models = train_final_models(data, chosen_methods)
    latest = data.iloc[-1]
    x_latest = latest[FEATURES].values.reshape(1, -1)

    p = {name: float(m.predict_proba(x_latest)[0, 1]) for name, m in models.items()}
    latest_regime = str(regimes.iloc[-1]) if len(regimes) else "UNKNOWN"
    weights_used = regime_w.get(latest_regime, global_w)
    ensemble_down = float(sum(weights_used.get(name, 0.0) * p[name] for name in p.keys()))
    signal = "BEARISH" if ensemble_down >= 0.60 else ("BULLISH" if ensemble_down < 0.45 else "NEUTRAL")
    backtest_stats = backtest(data, models, chosen_methods, global_w, regime_w, regimes)
    feature_snapshot = {k: float(latest[k]) for k in FEATURES}
    explainability = compute_global_importance(models, data)
    signal_change = compute_signal_change_diagnostics(feature_snapshot, ensemble_down, explainability, prev_report)

    joblib.dump(
        {
            "models": models,
            "features": FEATURES,
            "calibration": {"chosen_methods": chosen_methods, "summary": calib_summary},
            "ensemble": {
                "global_weights": global_w,
                "regime_weights": regime_w,
                "vol_thresh_21d": vol_thresh,
                "regime_definition": "vol_21d>=Q75 => HIGHVOL else LOWVOL; dev_ma_126>=0 => UPTREND else DOWNTREND",
            },
            "explainability": explainability,
        },
        ARTIFACT_DIR / "model_bundle.joblib",
    )

    # Reliability curve on the full in-sample fitted ensemble (diagnostic only)
    X_all = data[FEATURES].values
    y_all = data["target_down_21d"].values.astype(int)
    probs_all = np.mean([mdl.predict_proba(X_all)[:, 1] for mdl in models.values()], axis=0)
    reliability = build_reliability_curve(y_all, probs_all, n_bins=10)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "latest_date": str(pd.to_datetime(latest["date"]).date()),
        "latest_close": float(latest["close"]),
        "model_prob_down": p,
        "ensemble_down": ensemble_down,
        "ensemble_up": 1.0 - ensemble_down,
        "signal": signal,
        "thresholds": {"bearish": 0.60, "bullish": 0.45},
        "ensemble": {
            "type": "weighted_regime",
            "current_regime": latest_regime,
            "weights_used": weights_used,
            "global_weights": global_w,
            "regime_weights": regime_w,
            "vol_thresh_21d": vol_thresh,
        },
        "feature_snapshot": feature_snapshot,
        "validation": summarize_fold_stats(fold_stats),
        "calibration": {
            "per_model": calib_summary,
            "ensemble_reliability_curve": reliability,
        },
        "explainability": {
            **explainability,
            "signal_change": signal_change,
        },
        "backtest": backtest_stats,
    }
    (ARTIFACT_DIR / "latest_forecast.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved forecast to {ARTIFACT_DIR / 'latest_forecast.json'}")


if __name__ == "__main__":
    main()
