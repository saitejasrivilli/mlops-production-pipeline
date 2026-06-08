"""
Simulates a production data stream hitting the served model.
Injects controlled drift at step N to trigger the drift detector.
Shows: normal operation → data drift detected → alert fired → retrain triggered.

Run: python monitoring/simulate_production.py --n_steps 200 --drift_at 100
"""
import time
import argparse
import random
import json
import numpy as np
from pathlib import Path

from monitoring.drift_detector import DriftMonitor, compute_psi, _text_length_features


# ── Reference corpus (SST-2 style) ──────────────────────────────────────────
_REFERENCE_SENTENCES = [
    "This movie was absolutely fantastic, loved every minute of it",
    "Great film with excellent performances by the entire cast",
    "Terrible waste of time, I hated everything about it",
    "Boring and predictable, would not recommend this to anyone",
    "A masterpiece of cinema, beautifully crafted and deeply moving story",
    "The acting was superb and the plot kept me engaged throughout",
    "Disappointing sequel that fails to capture the magic of the original",
    "Outstanding direction and writing make this a must-watch film",
    "Poorly written script with cardboard characters and no real story arc",
    "A delightful surprise, far better than I expected from the reviews",
    "The special effects were impressive but the story felt hollow",
    "I laughed, I cried, this film hit every emotional note perfectly",
    "Slow pacing and confusing narrative made this hard to sit through",
    "Brilliant performances elevate an already strong screenplay",
    "The cinematography alone is worth watching this film for",
]

# ── Drifted corpus (technical/medical domain, very different from SST-2) ────
_DRIFTED_SENTENCES = [
    "The patient exhibited acute respiratory distress syndrome following intubation",
    "Quarterly earnings report indicates YoY revenue growth of 23.4 percent",
    "Algorithm complexity O(n log n) confirmed via empirical benchmarking on cluster",
    "Regulatory compliance framework updated per SEC filing requirements Q3",
    "Protein folding dynamics under pH 7.4 buffer conditions were analyzed",
    "The compiler optimization pass reduced binary size by approximately 18 percent",
    "Central bank monetary policy transmission mechanism analysis for emerging markets",
    "Genomic sequencing data from 1,200 cohort participants processed overnight",
    "Load balancer configuration updated for zero-downtime rolling deployment",
    "Clinical trial Phase III endpoints met with statistical significance p less than 0.001",
    "Transformer architecture attention head ablation study results presented at NeurIPS",
    "Supply chain disruption index rose 0.37 points amid port congestion data",
    "Kubernetes horizontal pod autoscaler triggered at 78 percent CPU utilization",
    "Multivariate regression coefficients significant at 95 percent confidence interval",
    "Database query execution plan optimized from 2.3 seconds to 180 milliseconds",
]


def generate_normal_batch(n: int = 32, seed: int = None) -> list[str]:
    """Generates typical SST-2-style sentences with natural length variation."""
    rng = random.Random(seed)
    sentences = []
    for _ in range(n):
        base = rng.choice(_REFERENCE_SENTENCES)
        # Occasionally append a short qualifier to vary length naturally
        if rng.random() < 0.3:
            qualifiers = [
                "overall",
                "in my opinion",
                "without a doubt",
                "to be honest",
                "all things considered",
            ]
            base = base + " " + rng.choice(qualifiers)
        sentences.append(base)
    return sentences


def generate_drifted_batch(n: int = 32, drift_type: str = "vocabulary") -> list[str]:
    """Generates out-of-distribution sentences (longer, different vocab, different domain).

    drift_type: 'vocabulary' — completely different domain (technical/medical)
                'length'     — extreme length shift (very short or very long)
    """
    rng = random.Random()
    sentences = []
    for _ in range(n):
        if drift_type == "length":
            # Mix of very long and very short sentences
            if rng.random() < 0.5:
                # Very short
                sentences.append(rng.choice(["good", "bad", "ok", "meh", "fine", "no"]))
            else:
                # Very long — repeat phrases
                base = rng.choice(_REFERENCE_SENTENCES)
                sentences.append(
                    base
                    + " and furthermore "
                    + base.lower()
                    + " which leads me to conclude that "
                    + base.lower()
                )
        else:
            # vocabulary drift — completely different domain
            sentences.append(rng.choice(_DRIFTED_SENTENCES))
    return sentences


def simulate(
    n_steps: int = 200,
    drift_at: int = 100,
    batch_size: int = 32,
    log_path: str = "logs/production_simulation.jsonl",
):
    """
    Main simulation loop:
    - Steps 0..drift_at: normal batches, monitor logs PSI < 0.1
    - Steps drift_at..: drifted batches, PSI crosses 0.2, alert fires
    - Prints timestamped log: step | psi | concept_drift | alert_fired
    - Saves full log to JSONL
    """
    rng = np.random.default_rng(0)

    # Build reference corpus for the monitor
    reference_data = generate_normal_batch(n=256, seed=42)
    monitor = DriftMonitor(
        reference_data=reference_data,
        accuracy_target=0.90,
        psi_threshold=0.20,
        cusum_slack=0.02,
        cusum_threshold=5.0,
    )

    # Pre-compute reference length stats for PSI replication in the sim output
    ref_lengths = _text_length_features(reference_data)
    ref_mean = float(ref_lengths.mean())
    ref_std = float(ref_lengths.std()) + 1e-6

    # Ensure log directory exists
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Production simulation starting — {n_steps} steps, drift injected at step {drift_at}")
    print(f"Reference corpus: {len(reference_data)} sentences, mean_len={ref_mean:.1f} words")
    print(f"Log → {log_path}\n")
    print(f"{'step':>6}  {'PSI':>8}  {'concept_drift':>14}  {'alert':>6}")
    print("-" * 46)

    log_records = []

    for step in range(n_steps):
        is_drifted = step >= drift_at

        if is_drifted:
            batch = generate_drifted_batch(n=batch_size, drift_type="vocabulary")
            # Simulate gradual accuracy drop after drift
            steps_after_drift = step - drift_at
            accuracy = max(0.70, 0.91 - steps_after_drift * 0.003 + rng.normal(0, 0.005))
        else:
            batch = generate_normal_batch(n=batch_size, seed=step)
            # Stable accuracy with minor noise
            accuracy = float(rng.normal(0.914, 0.008))
            accuracy = float(np.clip(accuracy, 0.88, 0.98))

        result = monitor.check(batch, float(accuracy))

        psi = result["psi"]
        concept_drift = result["concept_drift"]
        alert = result["data_drift"] or result["concept_drift"]

        marker = ""
        if step == 0:
            marker = "  <- normal"
        elif step == drift_at - 1:
            marker = "  <- last normal step"
        elif step == drift_at:
            marker = "  <- DRIFT INJECTED"
        elif alert and step == drift_at + 1:
            marker = "  <- alert firing"

        print(
            f"[step={step:03d}] PSI={psi:.3f} concept_drift={str(concept_drift):<5} "
            f"alert={str(alert):<5}{marker}"
        )

        record = {
            "step": step,
            "psi": round(psi, 4),
            "accuracy": round(float(accuracy), 4),
            "concept_drift": concept_drift,
            "data_drift": result["data_drift"],
            "alert": alert,
            "alert_message": result["alert_message"],
            "vocab_overlap": round(result["vocab_overlap"], 4),
            "bigram_jsd": round(result["bigram_jsd"], 4),
            "is_drifted_batch": is_drifted,
        }
        log_records.append(record)

        # Small sleep to simulate real-time stream (omit for speed in CI)
        # time.sleep(0.01)

    with open(log_file, "w") as f:
        for rec in log_records:
            f.write(json.dumps(rec) + "\n")

    # Summary statistics
    normal_psi = [r["psi"] for r in log_records if not r["is_drifted_batch"]]
    drifted_psi = [r["psi"] for r in log_records if r["is_drifted_batch"]]
    alerts_fired = sum(1 for r in log_records if r["alert"])

    print("\n" + "=" * 46)
    print("Simulation complete")
    print(f"  Normal steps  : {len(normal_psi)}  |  mean PSI={np.mean(normal_psi):.3f}")
    print(f"  Drifted steps : {len(drifted_psi)}  |  mean PSI={np.mean(drifted_psi):.3f}")
    print(f"  Alerts fired  : {alerts_fired}")
    print(f"  Log saved     : {log_path}")

    if drifted_psi and np.mean(drifted_psi) > 0.20:
        print("\n  ACTION: PSI > 0.20 threshold exceeded — retrain trigger would fire.")
        print("  In production: POST /api/retrain or dispatch GitHub Actions workflow.")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a production data stream with controlled drift injection."
    )
    parser.add_argument("--n_steps", type=int, default=200, help="Total simulation steps")
    parser.add_argument("--drift_at", type=int, default=100, help="Step at which drift is injected")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument(
        "--log_path",
        default="logs/production_simulation.jsonl",
        help="Path to write JSONL log",
    )
    args = parser.parse_args()
    simulate(
        n_steps=args.n_steps,
        drift_at=args.drift_at,
        batch_size=args.batch_size,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
