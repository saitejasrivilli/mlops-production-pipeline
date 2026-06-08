import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring.alerts import AlertManager, AlertRule, Severity, DEFAULT_RULES


class TestAlertManager:
    def setup_method(self):
        self.manager = AlertManager(dry_run=True, log_file=None)
        # Reset cooldowns
        for rule in self.manager.rules:
            rule._last_fired = 0.0

    def test_no_alerts_for_healthy_metrics(self):
        metrics = {"eval_accuracy": 0.93, "latency_p99_ms": 100.0, "psi": 0.05}
        fired = self.manager.check(metrics)
        assert fired == []

    def test_critical_alert_on_low_accuracy(self):
        metrics = {"eval_accuracy": 0.80}
        fired = self.manager.check(metrics)
        critical = [a for a in fired if a["severity"] == "CRITICAL"]
        assert len(critical) >= 1
        assert any("accuracy" in a["name"] for a in critical)

    def test_warning_alert_on_moderate_accuracy_drop(self):
        metrics = {"eval_accuracy": 0.87}
        fired = self.manager.check(metrics)
        warnings = [a for a in fired if a["severity"] == "WARNING"]
        assert len(warnings) >= 1

    def test_no_double_fire_during_cooldown(self):
        import time
        metrics = {"eval_accuracy": 0.80}
        fired1 = self.manager.check(metrics)
        fired2 = self.manager.check(metrics)
        # Second call should yield no new alerts (cooldown active)
        assert len(fired2) == 0

    def test_data_drift_alert(self):
        metrics = {"psi": 0.30}
        fired = self.manager.check(metrics)
        drift_alerts = [a for a in fired if "drift" in a["name"]]
        assert len(drift_alerts) >= 1

    def test_high_latency_alert(self):
        metrics = {"latency_p99_ms": 600.0}
        fired = self.manager.check(metrics)
        latency_alerts = [a for a in fired if "latency" in a["name"]]
        assert len(latency_alerts) >= 1

    def test_summary_counts(self):
        metrics = {"eval_accuracy": 0.80, "latency_p99_ms": 700.0, "psi": 0.25}
        fired = self.manager.check(metrics)
        summary = self.manager.summary()
        assert summary["total_alerts"] == len(fired)
        assert summary["critical"] + summary["warnings"] <= summary["total_alerts"]

    def test_missing_metric_skipped(self):
        metrics = {"some_other_metric": 100.0}
        fired = self.manager.check(metrics)
        assert fired == []


class TestAlertRule:
    def test_lt_operator(self):
        rule = AlertRule("r", "m", 0.90, "lt", Severity.CRITICAL, "{metric}={value}")
        assert rule.evaluate(0.89) is True
        assert rule.evaluate(0.91) is False

    def test_gt_operator(self):
        rule = AlertRule("r", "m", 500.0, "gt", Severity.WARNING, "{metric}={value}")
        assert rule.evaluate(600.0) is True
        assert rule.evaluate(400.0) is False

    def test_format_message(self):
        rule = AlertRule("r", "accuracy", 0.90, "lt", Severity.CRITICAL,
                         "{metric} = {value:.2f} < {threshold}")
        msg = rule.format_message(0.85)
        assert "0.85" in msg
        assert "0.9" in msg
