"""
Threshold-based alerting for ML pipeline health.

Supports: console, file, Slack webhook (optional), PagerDuty (optional).
"""
import json
import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import urllib.request


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertRule:
    name: str
    metric: str
    threshold: float
    operator: str          # "lt", "gt", "lte", "gte"
    severity: Severity
    message_template: str  # supports {metric}, {value}, {threshold}
    cooldown_seconds: int = 3600
    _last_fired: float = field(default=0.0, repr=False)

    def evaluate(self, value: float) -> bool:
        ops = {
            "lt": value < self.threshold,
            "gt": value > self.threshold,
            "lte": value <= self.threshold,
            "gte": value >= self.threshold,
        }
        return ops.get(self.operator, False)

    def format_message(self, value: float) -> str:
        return self.message_template.format(
            metric=self.metric, value=value, threshold=self.threshold
        )

    def is_cooling_down(self) -> bool:
        return (time.time() - self._last_fired) < self.cooldown_seconds

    def mark_fired(self):
        self._last_fired = time.time()


DEFAULT_RULES = [
    AlertRule(
        name="low_accuracy",
        metric="eval_accuracy",
        threshold=0.85,
        operator="lt",
        severity=Severity.CRITICAL,
        message_template="CRITICAL: {metric} dropped to {value:.4f} (threshold: {threshold}). Trigger retraining.",
        cooldown_seconds=3600,
    ),
    AlertRule(
        name="accuracy_warning",
        metric="eval_accuracy",
        threshold=0.88,
        operator="lt",
        severity=Severity.WARNING,
        message_template="WARNING: {metric} = {value:.4f} is approaching threshold {threshold}.",
        cooldown_seconds=1800,
    ),
    AlertRule(
        name="high_latency_p99",
        metric="latency_p99_ms",
        threshold=500.0,
        operator="gt",
        severity=Severity.WARNING,
        message_template="WARNING: {metric} = {value:.1f}ms exceeds {threshold}ms SLA.",
        cooldown_seconds=1800,
    ),
    AlertRule(
        name="data_drift",
        metric="psi",
        threshold=0.20,
        operator="gt",
        severity=Severity.CRITICAL,
        message_template="CRITICAL: Data drift detected! PSI = {value:.3f} > {threshold}.",
        cooldown_seconds=7200,
    ),
    AlertRule(
        name="concept_drift",
        metric="concept_drift_detected",
        threshold=0.5,
        operator="gt",
        severity=Severity.CRITICAL,
        message_template="CRITICAL: Concept drift detected at step {value}. Model accuracy degrading.",
        cooldown_seconds=7200,
    ),
]


class AlertManager:
    def __init__(
        self,
        rules: list = None,
        slack_webhook: str = None,
        log_file: str = "alerts.log",
        dry_run: bool = False,
    ):
        self.rules = rules or DEFAULT_RULES
        self.slack_webhook = slack_webhook or os.environ.get("SLACK_WEBHOOK_URL")
        self.log_file = log_file
        self.dry_run = dry_run
        self._fired_alerts = []

    def check(self, metrics: dict) -> list:
        fired = []
        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                continue
            if rule.evaluate(float(value)):
                if rule.is_cooling_down():
                    logger.debug(f"Alert '{rule.name}' in cooldown, skipping.")
                    continue
                msg = rule.format_message(float(value))
                alert = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "name": rule.name,
                    "severity": rule.severity.value,
                    "message": msg,
                    "metric": rule.metric,
                    "value": value,
                    "threshold": rule.threshold,
                }
                fired.append(alert)
                rule.mark_fired()
                self._dispatch(alert)
        self._fired_alerts.extend(fired)
        return fired

    def _dispatch(self, alert: dict):
        msg = f"[{alert['severity']}] {alert['timestamp']} — {alert['message']}"
        logger.warning(msg) if alert["severity"] != "INFO" else logger.info(msg)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(alert) + "\n")

        if self.slack_webhook and not self.dry_run:
            self._send_slack(alert)

    def _send_slack(self, alert: dict):
        emoji = {"CRITICAL": ":red_circle:", "WARNING": ":warning:", "INFO": ":white_circle:"}
        payload = {
            "text": f"{emoji.get(alert['severity'], '')} *{alert['severity']}* — {alert['message']}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*ML Pipeline Alert*\n{emoji.get(alert['severity'])} {alert['message']}\nMetric: `{alert['metric']}` = `{alert['value']}`",
                    },
                }
            ],
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.slack_webhook,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")

    def summary(self) -> dict:
        return {
            "total_alerts": len(self._fired_alerts),
            "critical": sum(1 for a in self._fired_alerts if a["severity"] == "CRITICAL"),
            "warnings": sum(1 for a in self._fired_alerts if a["severity"] == "WARNING"),
            "latest": self._fired_alerts[-1] if self._fired_alerts else None,
        }


if __name__ == "__main__":
    manager = AlertManager(dry_run=True)

    test_cases = [
        {"eval_accuracy": 0.84, "latency_p99_ms": 120.0, "psi": 0.05},
        {"eval_accuracy": 0.87, "latency_p99_ms": 600.0, "psi": 0.15},
        {"eval_accuracy": 0.82, "latency_p99_ms": 800.0, "psi": 0.25, "concept_drift_detected": 1},
    ]

    for i, metrics in enumerate(test_cases):
        print(f"\n--- Test case {i+1}: {metrics} ---")
        fired = manager.check(metrics)
        for alert in fired:
            print(f"  FIRED: {alert['message']}")

    print("\nSummary:", json.dumps(manager.summary(), indent=2))
