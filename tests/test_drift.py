import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring.drift_detector import compute_psi, cusum_detect, DriftMonitor


class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_very_different_distributions_high_psi(self):
        rng = np.random.default_rng(1)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(5, 1, 1000)   # mean shifted by 5 sigma
        psi = compute_psi(expected, actual)
        assert psi > 0.20, f"Expected PSI > 0.20 for large shift, got {psi:.3f}"

    def test_slight_distribution_shift(self):
        rng = np.random.default_rng(2)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(0.3, 1, 1000)   # small shift
        psi = compute_psi(expected, actual)
        assert psi < 0.20, f"Small shift should be < 0.20, got {psi:.3f}"

    def test_psi_non_negative(self):
        rng = np.random.default_rng(3)
        a = rng.uniform(0, 1, 500)
        b = rng.uniform(0.2, 1.2, 500)
        psi = compute_psi(a, b)
        assert psi >= 0

    def test_returns_float(self):
        data = np.arange(100, dtype=float)
        assert isinstance(compute_psi(data, data + 0.1), float)


class TestCUSUM:
    def test_no_drift_stable_accuracy(self):
        accuracy_series = [0.92, 0.91, 0.93, 0.90, 0.92, 0.91, 0.90, 0.92]
        result = cusum_detect(accuracy_series, target=0.90, slack=0.02, threshold=5.0)
        assert result["drift_detected"] is False

    def test_drift_detected_on_sharp_drop(self):
        accuracy_series = [0.92, 0.91, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70]
        result = cusum_detect(accuracy_series, target=0.90, slack=0.02, threshold=5.0)
        assert result["drift_detected"] is True
        assert result["drift_at_step"] is not None

    def test_drift_at_step_is_correct(self):
        # Accuracy is fine for steps 0-4, then drops hard
        accuracy_series = [0.92, 0.91, 0.90, 0.92, 0.91, 0.60, 0.60, 0.60, 0.60, 0.60]
        result = cusum_detect(accuracy_series, target=0.90, slack=0.02, threshold=5.0)
        assert result["drift_detected"] is True
        # Drift should be detected sometime after step 4
        assert result["drift_at_step"] >= 5

    def test_cusum_values_length_matches_input(self):
        series = [0.90, 0.89, 0.88, 0.87]
        result = cusum_detect(series, target=0.90, slack=0.02)
        assert len(result["cusum_values"]) == 4

    def test_cusum_values_non_negative(self):
        series = [0.85, 0.84, 0.83, 0.82, 0.81]
        result = cusum_detect(series, target=0.90)
        assert all(v >= 0 for v in result["cusum_values"])

    def test_empty_series(self):
        result = cusum_detect([], target=0.90)
        assert result["drift_detected"] is False
        assert result["drift_at_step"] is None
        assert result["cusum_values"] == []


class TestDriftMonitor:
    def setup_method(self):
        self.reference = [
            "The movie was excellent and well-crafted",
            "Terrible film, wasted two hours of my life",
            "Outstanding performances from the entire cast",
            "Boring and predictable storyline throughout",
            "A cinematic masterpiece worth watching twice",
        ] * 20

    def test_check_returns_expected_keys(self):
        monitor = DriftMonitor(self.reference, accuracy_target=0.90)
        result = monitor.check(self.reference, new_accuracy=0.91)
        expected_keys = {"data_drift", "concept_drift", "psi", "psi_threshold",
                         "cusum_stat", "cusum_threshold", "vocab_overlap",
                         "bigram_jsd", "accuracy_trend", "alert_message"}
        assert expected_keys.issubset(set(result.keys()))

    def test_no_drift_on_identical_data(self):
        monitor = DriftMonitor(self.reference, accuracy_target=0.90)
        result = monitor.check(self.reference, new_accuracy=0.92)
        assert result["data_drift"] is False
        assert result["concept_drift"] is False

    def test_concept_drift_on_accuracy_decline(self):
        monitor = DriftMonitor(self.reference, accuracy_target=0.90,
                               cusum_slack=0.01, cusum_threshold=2.0)
        accuracies = [0.75, 0.74, 0.73, 0.72, 0.71, 0.70, 0.69, 0.68]
        result = None
        for acc in accuracies:
            result = monitor.check(self.reference, new_accuracy=acc)
        assert result["concept_drift"] is True

    def test_accuracy_trend_accumulates(self):
        monitor = DriftMonitor(self.reference, accuracy_target=0.90)
        for acc in [0.91, 0.90, 0.89]:
            result = monitor.check(self.reference, new_accuracy=acc)
        assert result["accuracy_trend"] == [0.91, 0.90, 0.89]
