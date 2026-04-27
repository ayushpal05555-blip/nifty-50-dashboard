import { createServer } from 'node:http';
import { appendFile, readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = __filename.slice(0, __filename.lastIndexOf('/'));
const DASHBOARD_FILE = join(__dirname, 'nifty50_live_dashboard (2).html');
const FORECAST_FILE = join(__dirname, 'ml', 'artifacts', 'latest_forecast.json');
const TRAIN_SCRIPT = join(__dirname, 'ml', 'train_pipeline.py');
const PACKAGE_FILE = join(__dirname, 'package.json');
const LOG_FILE = join(__dirname, 'ml', 'artifacts', 'retrain_runs.jsonl');
const PORT = Number(process.env.PORT || 8787);
const RETRAIN_TIMEOUT_MS = Number(process.env.RETRAIN_TIMEOUT_MS || 15 * 60 * 1000);
const FORECAST_STALE_HOURS = Number(process.env.FORECAST_STALE_HOURS || 72);

const IST_TZ = 'Asia/Kolkata';

function formatIstDateTitle(date) {
  return new Intl.DateTimeFormat('en-IN', {
    timeZone: IST_TZ,
    day: '2-digit',
    month: 'long',
    year: 'numeric'
  }).format(date);
}

function toIstMonthYear(unixSec) {
  const d = new Date(unixSec * 1000);
  const parts = new Intl.DateTimeFormat('en-IN', {
    timeZone: IST_TZ,
    year: 'numeric',
    month: 'numeric'
  }).formatToParts(d);
  const pick = (type) => Number((parts.find((p) => p.type === type) || {}).value || 0);
  return { year: pick('year'), month: pick('month') };
}

async function fetchYahooChart(symbol) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=3mo&interval=1d`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) throw new Error(`Yahoo fetch failed for ${symbol}`);
  const json = await res.json();
  const result = json?.chart?.result?.[0];
  if (!result) throw new Error(`No chart data for ${symbol}`);
  const closes = result?.indicators?.quote?.[0]?.close || [];
  const timestamps = result?.timestamp || [];
  const meta = result?.meta || {};
  return { closes, timestamps, meta };
}

function computeMtdFromSeries(timestamps, closes) {
  const now = new Date();
  const nowYear = Number(new Intl.DateTimeFormat('en-IN', { timeZone: IST_TZ, year: 'numeric' }).format(now));
  const nowMonth = Number(new Intl.DateTimeFormat('en-IN', { timeZone: IST_TZ, month: 'numeric' }).format(now));

  const pairs = timestamps
    .map((ts, i) => ({ ts, close: closes[i] }))
    .filter((p) => Number.isFinite(p.close));

  const thisMonth = pairs.filter((p) => {
    const ym = toIstMonthYear(p.ts);
    return ym.year === nowYear && ym.month === nowMonth;
  });

  if (thisMonth.length < 2) return null;
  const first = thisMonth[0].close;
  const last = thisMonth[thisMonth.length - 1].close;
  if (!Number.isFinite(first) || !Number.isFinite(last) || first === 0) return null;
  return ((last / first) - 1) * 100;
}

async function getLiveSnapshot() {
  const [niftyData, vixData] = await Promise.all([
    fetchYahooChart('^NSEI'),
    fetchYahooChart('^INDIAVIX')
  ]);

  const nifty = Number(niftyData.meta?.regularMarketPrice) || Number(niftyData.closes.filter(Number.isFinite).slice(-1)[0]);
  const vix = Number(vixData.meta?.regularMarketPrice) || Number(vixData.closes.filter(Number.isFinite).slice(-1)[0]);
  const mtd = computeMtdFromSeries(niftyData.timestamps, niftyData.closes);

  if (!Number.isFinite(nifty) || !Number.isFinite(vix) || !Number.isFinite(mtd)) {
    throw new Error('Invalid live snapshot values');
  }

  return {
    ok: true,
    source: 'YAHOO SERVER',
    nifty: Number(nifty.toFixed(2)),
    vix: Number(vix.toFixed(2)),
    mtd: Number(mtd.toFixed(2)),
    marketDate: formatIstDateTitle(new Date())
  };
}

function json(res, code, payload) {
  res.writeHead(code, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store'
  });
  res.end(JSON.stringify(payload));
}

let retrainRunning = false;
let retrainLast = null;
let serverStartedAt = new Date().toISOString();
let appMeta = { name: 'nifty50-live-dashboard-server', version: 'unknown' };

async function initAppMeta() {
  try {
    const raw = await readFile(PACKAGE_FILE, 'utf8');
    const pkg = JSON.parse(raw);
    appMeta = { name: String(pkg.name || appMeta.name), version: String(pkg.version || appMeta.version) };
  } catch {
    // Keep fallback metadata on read errors.
  }
}

function requiredForecastFields(parsed) {
  const required = [
    'generated_at',
    'latest_date',
    'latest_close',
    'ensemble_down',
    'ensemble_up',
    'signal',
    'model_prob_down',
    'feature_snapshot',
    'validation',
    'backtest'
  ];
  const missing = required.filter((k) => !(k in parsed));
  return { ok: missing.length === 0, missing };
}

function forecastFreshness(generatedAtIso) {
  const ts = Date.parse(generatedAtIso || '');
  if (!Number.isFinite(ts)) return { isFresh: false, ageHours: null };
  const ageMs = Date.now() - ts;
  const ageHours = ageMs / (1000 * 60 * 60);
  return { isFresh: ageHours <= FORECAST_STALE_HOURS, ageHours: Number(ageHours.toFixed(2)) };
}

async function writeStructuredLog(entry) {
  const line = JSON.stringify({ ts: new Date().toISOString(), ...entry }) + '\n';
  try {
    await appendFile(LOG_FILE, line, 'utf8');
  } catch {
    // Best effort only.
  }
}

function addCheck(checks, alerts, check) {
  checks.push(check);
  if (check.level !== 'OK') alerts.push(check);
}

function evaluateMonitoring(forecast) {
  const checks = [];
  const alerts = [];

  if (!forecast?.ok) {
    const failed = {
      id: 'forecast_artifact',
      level: 'CRITICAL',
      message: 'Forecast artifact unavailable or invalid',
      value: forecast?.error || forecast?.source || 'unknown'
    };
    addCheck(checks, alerts, failed);
    return {
      status: 'CRITICAL',
      checks,
      alerts,
      retrain_trigger: true,
      rollback_recommended: true,
      reasons: ['forecast_missing_or_invalid']
    };
  }

  const modelProbs = forecast.model_prob_down || {};
  const modelVals = [modelProbs.logreg, modelProbs.rf, modelProbs.gb].filter(Number.isFinite);
  const maxProb = modelVals.length ? Math.max(...modelVals) : null;
  const minProb = modelVals.length ? Math.min(...modelVals) : null;
  const modelSpread = Number.isFinite(maxProb) && Number.isFinite(minProb) ? Number((maxProb - minProb).toFixed(4)) : null;

  const briers = ['logreg', 'rf', 'gb']
    .map((m) => forecast.validation?.[m]?.brier_mean)
    .filter(Number.isFinite);
  const worstBrier = briers.length ? Math.max(...briers) : null;

  const vol21 = forecast.feature_snapshot?.vol_21d;
  const vixNorm = forecast.feature_snapshot?.vix_norm_21d;
  const signalDeltaPp = forecast.explainability?.signal_change?.delta_ensemble_down_pp;

  addCheck(checks, alerts, {
    id: 'artifact_freshness',
    level: forecast._meta?.fresh ? 'OK' : 'WARN',
    message: forecast._meta?.fresh ? 'Forecast artifact is fresh' : 'Forecast artifact is stale',
    value: forecast._meta?.age_hours
  });

  addCheck(checks, alerts, {
    id: 'calibration_brier',
    level: Number.isFinite(worstBrier) && worstBrier <= 0.25 ? 'OK' : (Number.isFinite(worstBrier) ? 'WARN' : 'CRITICAL'),
    message: Number.isFinite(worstBrier)
      ? `Worst model Brier=${worstBrier.toFixed(3)}`
      : 'Brier metrics missing',
    value: worstBrier
  });

  addCheck(checks, alerts, {
    id: 'model_disagreement',
    level: Number.isFinite(modelSpread) && modelSpread <= 0.12 ? 'OK' : (Number.isFinite(modelSpread) ? 'WARN' : 'CRITICAL'),
    message: Number.isFinite(modelSpread)
      ? `Inter-model probability spread=${modelSpread.toFixed(3)}`
      : 'Model probability spread unavailable',
    value: modelSpread
  });

  addCheck(checks, alerts, {
    id: 'data_drift_proxy_volatility',
    level: Number.isFinite(vol21) && vol21 <= 0.08 ? 'OK' : (Number.isFinite(vol21) ? 'WARN' : 'CRITICAL'),
    message: Number.isFinite(vol21) ? `vol_21d=${vol21.toFixed(4)}` : 'vol_21d missing',
    value: vol21
  });

  addCheck(checks, alerts, {
    id: 'data_drift_proxy_vix_norm',
    level: Number.isFinite(vixNorm) && Math.abs(vixNorm) <= 2.0 ? 'OK' : (Number.isFinite(vixNorm) ? 'WARN' : 'CRITICAL'),
    message: Number.isFinite(vixNorm) ? `vix_norm_21d=${vixNorm.toFixed(3)}` : 'vix_norm_21d missing',
    value: vixNorm
  });

  addCheck(checks, alerts, {
    id: 'signal_stability',
    level: Number.isFinite(signalDeltaPp) && Math.abs(signalDeltaPp) <= 8 ? 'OK' : (Number.isFinite(signalDeltaPp) ? 'WARN' : 'OK'),
    message: Number.isFinite(signalDeltaPp)
      ? `Signal delta=${signalDeltaPp.toFixed(2)}pp`
      : 'No previous signal for stability comparison',
    value: signalDeltaPp
  });

  const liveAccuracyWindow = forecast.live_monitoring?.window_days;
  const liveAccuracy = forecast.live_monitoring?.hit_rate;
  addCheck(checks, alerts, {
    id: 'live_accuracy',
    level: Number.isFinite(liveAccuracy) ? (liveAccuracy >= 0.52 ? 'OK' : 'WARN') : 'WARN',
    message: Number.isFinite(liveAccuracy)
      ? `Live hit-rate=${(liveAccuracy * 100).toFixed(1)}% over ${liveAccuracyWindow || 'n/a'} days`
      : 'Live accuracy feed missing (add live_monitoring.hit_rate)',
    value: Number.isFinite(liveAccuracy) ? liveAccuracy : null
  });

  const critical = alerts.some((a) => a.level === 'CRITICAL');
  const warnCount = alerts.filter((a) => a.level === 'WARN').length;
  const status = critical ? 'CRITICAL' : (warnCount >= 2 ? 'WARN' : 'OK');
  const retrainTrigger = critical || warnCount >= 2;
  const rollbackRecommended = Boolean(critical || (retrainLast && retrainLast.ok === false));

  const reasons = [];
  if (!forecast._meta?.fresh) reasons.push('stale_forecast');
  if (Number.isFinite(worstBrier) && worstBrier > 0.25) reasons.push('brier_drift');
  if (Number.isFinite(modelSpread) && modelSpread > 0.12) reasons.push('model_disagreement');
  if (Number.isFinite(vol21) && vol21 > 0.08) reasons.push('volatility_drift');
  if (Number.isFinite(vixNorm) && Math.abs(vixNorm) > 2.0) reasons.push('vix_regime_shift');
  if (Number.isFinite(signalDeltaPp) && Math.abs(signalDeltaPp) > 8) reasons.push('signal_instability');
  if (!Number.isFinite(liveAccuracy)) reasons.push('live_accuracy_missing');
  if (Number.isFinite(liveAccuracy) && liveAccuracy < 0.52) reasons.push('live_accuracy_drop');
  if (retrainLast && retrainLast.ok === false) reasons.push('last_retrain_failed');

  return {
    status,
    checks,
    alerts,
    retrain_trigger: retrainTrigger,
    rollback_recommended: rollbackRecommended,
    reasons
  };
}

function runRetrainJob() {
  return new Promise((resolve) => {
    const py = spawn('python3', [TRAIN_SCRIPT], {
      cwd: __dirname,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';
    py.stdout.on('data', (buf) => {
      stdout += String(buf);
    });
    py.stderr.on('data', (buf) => {
      stderr += String(buf);
    });
    const timeout = setTimeout(() => {
      try {
        py.kill('SIGTERM');
      } catch {
        // ignore kill errors
      }
      resolve({
        ok: false,
        code: -2,
        stdout: stdout.trim(),
        stderr: `Timed out after ${RETRAIN_TIMEOUT_MS}ms`
      });
    }, RETRAIN_TIMEOUT_MS);

    py.on('close', (code) => {
      clearTimeout(timeout);
      resolve({
        ok: code === 0,
        code,
        stdout: stdout.trim(),
        stderr: stderr.trim()
      });
    });
    py.on('error', (err) => {
      clearTimeout(timeout);
      resolve({
        ok: false,
        code: -1,
        stdout: '',
        stderr: String(err?.message || err)
      });
    });
  });
}

async function getForecastSnapshot() {
  try {
    const raw = await readFile(FORECAST_FILE, 'utf8');
    const parsed = JSON.parse(raw);
    const shape = requiredForecastFields(parsed);
    const freshness = forecastFreshness(parsed.generated_at);

    if (!shape.ok) {
      return {
        ok: false,
        source: 'FORECAST INVALID',
        error: 'Missing required fields',
        missing_fields: shape.missing
      };
    }

    return {
      ok: true,
      ...parsed,
      _meta: {
        fresh: freshness.isFresh,
        age_hours: freshness.ageHours
      }
    };
  } catch {
    return {
      ok: false,
      source: 'FORECAST UNAVAILABLE'
    };
  }
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || '/', `http://${req.headers.host}`);
    if (url.pathname === '/api/live') {
      try {
        const data = await getLiveSnapshot();
        return json(res, 200, data);
      } catch {
        return json(res, 200, {
          ok: false,
          source: 'STATIC FALLBACK'
        });
      }
    }

    if (url.pathname === '/api/forecast') {
      const forecast = await getForecastSnapshot();
      return json(res, 200, {
        ...forecast,
        api_meta: {
          app_name: appMeta.name,
          app_version: appMeta.version,
          served_at: new Date().toISOString()
        }
      });
    }

    if (url.pathname === '/api/retrain') {
      if (req.method !== 'POST' && req.method !== 'GET') {
        return json(res, 405, { ok: false, error: 'Method not allowed' });
      }
      if (retrainRunning) {
        return json(res, 409, { ok: false, error: 'Retrain already running' });
      }
      retrainRunning = true;
      const startedAtMs = Date.now();
      const out = await runRetrainJob();
      retrainRunning = false;
      retrainLast = {
        ok: out.ok,
        code: out.code,
        duration_ms: Date.now() - startedAtMs,
        at: new Date().toISOString(),
        stderr: out.stderr || null
      };
      await writeStructuredLog({
        event: 'retrain',
        ok: out.ok,
        code: out.code,
        duration_ms: retrainLast.duration_ms,
        stderr: out.stderr || null
      });
      if (!out.ok) {
        return json(res, 500, {
          ok: false,
          error: 'Retrain failed',
          code: out.code,
          stderr: out.stderr || 'No stderr'
        });
      }
      return json(res, 200, {
        ok: true,
        message: 'Retrain completed',
        code: out.code,
        stdout: out.stdout || 'No stdout',
        duration_ms: retrainLast.duration_ms
      });
    }

    if (url.pathname === '/api/health') {
      const forecast = await getForecastSnapshot();
      const monitoring = evaluateMonitoring(forecast);
      return json(res, 200, {
        ok: true,
        status: monitoring.status === 'CRITICAL' ? 'DEGRADED' : (forecast.ok && forecast._meta?.fresh !== false ? 'UP' : 'DEGRADED'),
        time: new Date().toISOString(),
        uptime_sec: Math.floor(process.uptime()),
        server_started_at: serverStartedAt,
        app: {
          name: appMeta.name,
          version: appMeta.version
        },
        retrain: {
          running: retrainRunning,
          timeout_ms: RETRAIN_TIMEOUT_MS,
          last: retrainLast
        },
        forecast: {
          available: Boolean(forecast.ok),
          fresh: forecast._meta?.fresh ?? false,
          age_hours: forecast._meta?.age_hours ?? null,
          latest_date: forecast.latest_date || null,
          generated_at: forecast.generated_at || null,
          validation_error: forecast.ok ? null : forecast.error || 'forecast unavailable'
        },
        monitoring: {
          status: monitoring.status,
          retrain_trigger: monitoring.retrain_trigger,
          rollback_recommended: monitoring.rollback_recommended,
          alert_count: monitoring.alerts.length
        }
      });
    }

    if (url.pathname === '/api/monitoring') {
      const forecast = await getForecastSnapshot();
      const monitoring = evaluateMonitoring(forecast);
      return json(res, 200, {
        ok: true,
        time: new Date().toISOString(),
        app: {
          name: appMeta.name,
          version: appMeta.version
        },
        monitoring,
        policy: {
          retrain_trigger_when: 'status=CRITICAL or >=2 WARN alerts',
          rollback_policy: [
            'Rollback immediately if new retrain artifact is invalid/missing required fields.',
            'Rollback if post-retrain CRITICAL alerts increase vs pre-retrain snapshot.',
            'Rollback if live_monitoring.hit_rate drops below 0.48 for 3 consecutive checks.',
            'Fallback target is last known good model_bundle.joblib + latest_forecast.json pair.'
          ]
        },
        last_retrain: retrainLast
      });
    }

    if (url.pathname === '/' || url.pathname === '/dashboard') {
      const html = await readFile(DASHBOARD_FILE, 'utf8');
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(html);
      return;
    }

    const isHtml = extname(url.pathname).toLowerCase() === '.html';
    if (isHtml) {
      const html = await readFile(DASHBOARD_FILE, 'utf8');
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(html);
      return;
    }

    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  } catch {
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Internal server error');
  }
});

await initAppMeta();
serverStartedAt = new Date().toISOString();
server.listen(PORT, () => {
  console.log(`Dashboard server running at http://localhost:${PORT} (${appMeta.name}@${appMeta.version})`);
});
