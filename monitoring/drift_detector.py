"""
Detects two types of drift:
1. Data drift: distribution shift in input text (length, vocabulary, embedding distance)
2. Concept drift: model accuracy drop over time

Uses Population Stability Index (PSI) for data drift.
Uses CUSUM (cumulative sum) for concept drift.

PSI interpretation:
  < 0.10  — No significant change
  0.10–0.20 — Slight change, monitor
  > 0.20  — Significant change, investigate / retrain
"""
import json
import argparse
import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict
from scipy import stats
from typing import Optional


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions. PSI > 0.2 = significant drift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Build bins from expected
    breakpoints = np.linspace(expected.min(), expected.max(), n_bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Smooth zeros to avoid log(0)
    expected_pct = np.where(expected_counts == 0, 1e-4, expected_counts) / len(expected)
    actual_pct = np.where(actual_counts == 0, 1e-4, actual_counts) / len(actual)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def cusum_detect(
    accuracy_series: list,
    target: float,
    slack: float = 0.02,
    threshold: float = 5.0,
) -> dict:
    """
    CUSUM change-point detection for downward accuracy drift.

    Detects when accuracy falls below `target - slack` persistently.
    threshold: cumulative sum threshold to declare drift.

    Returns {"drift_detected": bool, "drift_at_step": int | None, "cusum_values": list}
    """
    cusum = 0.0
    cusum_values = []
    drift_at_step = None

    for i, acc in enumerate(accuracy_series):
        # Detect downward drift: negative contribution when acc < target - slack
        cusum = max(0.0, cusum + (target - slack - acc))
        cusum_values.append(round(cusum, 4))
        if cusum > threshold and drift_at_step is None:
            drift_at_step = i

    return {
        "drift_detected": drift_at_step is not None,
        "drift_at_step": drift_at_step,
        "cusum_values": cusum_values,
        "final_cusum": cusum_values[-1] if cusum_values else 0.0,
    }


def _text_length_features(texts: list) -> np.ndarray:
    return np.array([len(t.split()) for t in texts], dtype=float)


def _vocab_overlap(ref_texts: list, new_texts: list) -> float:
    ref_vocab = set(w for t in ref_texts for w in t.lower().split())
    new_vocab = set(w for t in new_texts for w in t.lower().split())
    if not ref_vocab:
        return 1.0
    return len(new_vocab & ref_vocab) / len(ref_vocab | new_vocab)


def _top_bigram_shift(ref_texts: list, new_texts: list, top_k: int = 50) -> float:
    def bigrams(texts):
        bg = Counter()
        for t in texts:
            words = t.lower().split()
            for i in range(len(words) - 1):
                bg[(words[i], words[i + 1])] += 1
        return bg

    ref_bg = bigrams(ref_texts)
    new_bg = bigrams(new_texts)
    top = set(k for k, _ in ref_bg.most_common(top_k))
    if not top:
        return 0.0
    ref_vec = np.array([ref_bg[k] for k in top], dtype=float)
    new_vec = np.array([new_bg[k] for k in top], dtype=float)
    ref_vec = ref_vec / (ref_vec.sum() + 1e-9)
    new_vec = new_vec / (new_vec.sum() + 1e-9)
    # Jensen-Shannon divergence
    m = (ref_vec + new_vec) / 2
    js = 0.5 * (stats.entropy(ref_vec + 1e-9, m + 1e-9) + stats.entropy(new_vec + 1e-9, m + 1e-9))
    return float(js)


@dataclass
class DriftReport:
    data_drift: bool
    concept_drift: bool
    psi: float
    psi_threshold: float
    cusum_stat: float
    cusum_threshold: float
    vocab_overlap: float
    bigram_jsd: float
    accuracy_trend: list
    alert_message: str


class DriftMonitor:
    def __init__(
        self,
        reference_data: list,
        accuracy_target: float = 0.90,
        psi_threshold: float = 0.20,
        cusum_slack: float = 0.02,
        cusum_threshold: float = 5.0,
    ):
        self.reference_data = reference_data
        self.accuracy_target = accuracy_target
        self.psi_threshold = psi_threshold
        self.cusum_slack = cusum_slack
        self.cusum_threshold = cusum_threshold
        self._accuracy_history = []
        self._ref_lengths = _text_length_features(reference_data)

    def check(self, new_data: list, new_accuracy: float) -> dict:
        self._accuracy_history.append(new_accuracy)

        new_lengths = _text_length_features(new_data)
        psi = compute_psi(self._ref_lengths, new_lengths)

        cusum_result = cusum_detect(
            self._accuracy_history,
            target=self.accuracy_target,
            slack=self.cusum_slack,
            threshold=self.cusum_threshold,
        )

        vocab_ov = _vocab_overlap(self.reference_data, new_data)
        bigram_jsd = _top_bigram_shift(self.reference_data, new_data)

        data_drift = psi > self.psi_threshold
        concept_drift = cusum_result["drift_detected"]

        messages = []
        if data_drift:
            messages.append(f"DATA DRIFT: PSI={psi:.3f} > {self.psi_threshold}")
        if concept_drift:
            messages.append(
                f"CONCEPT DRIFT: accuracy dropped below target at step {cusum_result['drift_at_step']}"
            )
        alert = " | ".join(messages) if messages else "No drift detected."

        report = DriftReport(
            data_drift=data_drift,
            concept_drift=concept_drift,
            psi=psi,
            psi_threshold=self.psi_threshold,
            cusum_stat=cusum_result["final_cusum"],
            cusum_threshold=self.cusum_threshold,
            vocab_overlap=vocab_ov,
            bigram_jsd=bigram_jsd,
            accuracy_trend=self._accuracy_history.copy(),
            alert_message=alert,
        )
        return asdict(report)


def demo():
    rng = np.random.default_rng(42)

    reference = [
        "This movie was absolutely fantastic, loved every minute",
        "Great film with excellent performances by the cast",
        "Terrible waste of time, I hated everything about it",
        "Boring and predictable, would not recommend to anyone",
        "A masterpiece of cinema, beautifully crafted story",
    ] * 40

    # Simulate gradual data drift: shorter, more informal text
    drifted = [
        "good",
        "bad",
        "ok film",
        "meh",
        "loved it",
        "hated it",
        "5 stars",
        "1 star",
    ] * 25

    monitor = DriftMonitor(reference_data=reference, accuracy_target=0.90)

    # Simulate 10 evaluation rounds with declining accuracy
    accuracy_series = [0.92, 0.91, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.75]

    print("=== Drift Monitor Demo ===\n")
    for i, acc in enumerate(accuracy_series):
        result = monitor.check(drifted, acc)
        print(f"Round {i+1:2d} | accuracy={acc:.2f} | PSI={result['psi']:.3f} | "
              f"CUSUM={result['cusum_stat']:.2f} | {result['alert_message']}")

    print("\nFull report (last round):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", default=True)
    args = parser.parse_args()
    if args.demo:
        demo()
