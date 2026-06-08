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

### `pipeline/feature_store.py`
Redis-backed feature store with automatic in-memory fallback. Implements the cache-aside pattern for precomputed text embeddings.

```python
from pipeline.feature_store import FeatureStore, EmbeddingStore

# Feature store — falls back to in-memory dict if Redis is unavailable
store = FeatureStore(redis_url="redis://localhost:6379", ttl=3600)
store.put("user:1234", {"age_bucket": 3, "country": "US", "segment": "premium"})
features = store.get("user:1234")

# Batch operations
records = [{"id": f"item:{i}", "price": i * 1.5, "category": "electronics"} for i in range(100)]
store.put_batch(records, key_field="id")
batch = store.get_batch([f"item:{i}" for i in range(5)])

# Cache-aside: compute embedding only on miss
embed_store = EmbeddingStore()
vec = embed_store.get_or_compute("sentiment text here", compute_fn=my_embed_fn)

print(store.stats())
# {'backend': 'redis', 'n_keys': 101, 'hit_rate': 0.833, 'avg_latency_ms': 0.41, ...}
```

**Cache-aside pattern:** `get_or_compute(text, fn)` checks the cache first; on a miss it calls `fn(text)`, stores the result under `sha256(text)[:16]`, and returns it. This keeps expensive embedding computation out of the hot path and makes latency predictable under load.

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

### `monitoring/simulate_production.py`
End-to-end production stream simulator. Injects controlled vocabulary drift at a configurable step to demonstrate the full detection pipeline: normal operation → PSI spike → alert fired → retrain trigger.

```bash
python monitoring/simulate_production.py --n_steps 200 --drift_at 100
```

Console output:
```
[step=000] PSI=0.023 concept_drift=False alert=False  <- normal
[step=099] PSI=0.087 concept_drift=False alert=False
[step=100] PSI=0.234 concept_drift=False alert=True   <- DRIFT INJECTED
[step=101] PSI=0.271 concept_drift=False alert=True   <- alert firing
...
Simulation complete
  Normal steps  : 100  |  mean PSI=0.061
  Drifted steps : 100  |  mean PSI=0.312
  Alerts fired  : 100
  ACTION: PSI > 0.20 threshold exceeded — retrain trigger would fire.
```

The simulator uses two corpora: an SST-2-style sentiment corpus (normal) and a technical/medical domain corpus (drifted). At step `drift_at`, batches switch to the drifted corpus. The `DriftMonitor` detects the distribution shift via PSI on text-length distributions and bigram Jensen-Shannon divergence. All per-step records are written to `logs/production_simulation.jsonl` for offline analysis.

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
