# MLOps Production Pipeline

End-to-end ML CI/CD system for text classification: automated training, evaluation, model registration, serving, and drift-aware monitoring — all orchestrated via GitHub Actions.

## Architecture

```
Code Push
    │
    ▼
GitHub Actions CI/CD
    │
    ├─── [lint-and-test] ──── ruff + pytest
    │
    ├─── [train] ──────────── DistilBERT on SST-2
    │         │                MLflow: log params, metrics, artifacts
    │         ▼
    ├─── [evaluate] ───────── accuracy, F1, latency P50/P99
    │         │
    │         ▼
    ├─── [register] ───────── promote to Staging if accuracy > 0.90
    │                         MLflow Model Registry
    │         │
    │         ▼
    └─── [serve] ──────────── BentoML REST API
                               POST /classify
                               POST /batch_classify


Drift Monitor (weekly cron + on-demand)
    │
    ├─── Data Drift ───────── PSI on text length distribution
    │                         Vocabulary overlap
    │                         Bigram Jensen-Shannon divergence
    │
    └─── Concept Drift ─────  CUSUM on rolling accuracy
              │
              ▼
         Alert Manager ────── Slack / PagerDuty / log
              │
              ▼
         Retrain trigger ───── GitHub Actions workflow_dispatch
```

## Components

### `pipeline/train.py`
Fine-tunes DistilBERT on SST-2 (binary sentiment), logs all metrics and the model artifact to MLflow.

```bash
python pipeline/train.py --run_name my_run --num_epochs 1 --max_samples 1000
```

### `pipeline/evaluate.py`
Loads model from MLflow run, evaluates accuracy/F1 and measures per-sample latency at P50/P99.

```bash
python pipeline/evaluate.py --run_id <RUN_ID>
```

### `pipeline/register.py`
Promotes a run's model to MLflow Model Registry `Staging` stage only if it beats the accuracy threshold **and** the current Staging model.

```bash
python pipeline/register.py --threshold 0.90
```

### `pipeline/serve.py`
BentoML service wrapping the registered model.

```bash
python pipeline/serve.py --save          # import from MLflow → BentoML
bentoml serve pipeline/serve.py:svc      # serve on :3000
```

Endpoints:
| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/classify` | `{"text": "..."}` | `{"label": "positive", "confidence": 0.97}` |
| POST | `/batch_classify` | `[{"text": "..."}, ...]` | list of results |
| GET | `/health` | — | `{"status": "ok"}` |

### `monitoring/drift_detector.py`
Two-signal drift detection:

| Signal | Method | Threshold |
|--------|--------|-----------|
| Data drift | Population Stability Index (PSI) on text features | PSI > 0.20 = significant |
| Concept drift | CUSUM on rolling accuracy series | cumsum > 5.0 = triggered |

PSI interpretation:
- `< 0.10` — no change
- `0.10–0.20` — monitor closely
- `> 0.20` — retrain

```bash
python monitoring/drift_detector.py --demo
```

### `monitoring/alerts.py`
Rule-based alerting with cooldown logic. Supports Slack webhook (set `SLACK_WEBHOOK_URL` env var).

Default rules:
| Rule | Condition | Severity |
|------|-----------|----------|
| `low_accuracy` | accuracy < 0.85 | CRITICAL |
| `accuracy_warning` | accuracy < 0.88 | WARNING |
| `high_latency_p99` | P99 latency > 500ms | WARNING |
| `data_drift` | PSI > 0.20 | CRITICAL |
| `concept_drift` | CUSUM triggered | CRITICAL |

## CI/CD Workflow

`.github/workflows/ml_pipeline.yml` defines 5 jobs:

1. **lint-and-test** — ruff linting + pytest unit tests
2. **train** — trains model, stores run ID as artifact
3. **evaluate** — loads model from MLflow, computes final metrics
4. **register** — promotes to Staging if threshold met (main branch only)
5. **drift-check** — PSI + CUSUM + alerts
6. **notify** — Slack notification on failure

Triggers: push to `main`, PRs, **weekly cron (Monday 02:00 UTC)**, manual dispatch.

## Setup

```bash
pip install -r requirements.txt

# Start MLflow tracking server (local)
mlflow server --host 0.0.0.0 --port 5000

# Run full pipeline manually
python pipeline/train.py --run_name baseline
python pipeline/evaluate.py
python pipeline/register.py --threshold 0.90
python pipeline/serve.py --save
bentoml serve pipeline/serve.py:svc
```

For GitHub Actions, set these secrets:
- `MLFLOW_TRACKING_URI` — your MLflow server URL
- `SLACK_WEBHOOK_URL` — (optional) Slack incoming webhook

## Tests

```bash
pytest tests/ -v
```

Tests cover PSI edge cases, CUSUM drift detection correctness, alert rule evaluation, and cooldown behavior.

## Key Design Choices

- **MLflow** for experiment tracking and model registry — single source of truth for all runs.
- **BentoML** for serving — production-grade, supports batching, async runners, Docker export.
- **PSI** for data drift — standard in production ML systems (credit scoring, ad ranking); interpretable thresholds.
- **CUSUM** for concept drift — detects persistent accuracy decline before it becomes catastrophic, unlike simple threshold checks that miss gradual degradation.
- **Paired promotion** — new models only replace current Staging if they're strictly better, preventing regressions from noisy CI runs.
