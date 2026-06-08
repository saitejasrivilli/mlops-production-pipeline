import argparse
import time
import mlflow
import mlflow.pytorch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate(
    run_id: str = None,
    model_uri: str = None,
    eval_split: str = "validation",
    dataset_name: str = "sst2",
    batch_size: int = 32,
    max_samples: int = 500,
) -> dict:
    if run_id is None and model_uri is None:
        try:
            with open("last_run_id.txt") as f:
                run_id = f.read().strip()
        except FileNotFoundError:
            raise ValueError("Provide --run_id or --model_uri, or run train.py first.")

    if model_uri is None:
        model_uri = f"runs:/{run_id}/model"

    print(f"Loading model from: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    run = mlflow.get_run(run_id) if run_id else None
    model_name_for_tok = "distilbert-base-uncased"
    if run:
        model_name_for_tok = run.data.params.get("model_name", model_name_for_tok)

    tokenizer = AutoTokenizer.from_pretrained(model_name_for_tok)
    ds = load_dataset("glue", dataset_name)
    eval_ds = ds[eval_split].select(range(min(max_samples, len(ds[eval_split]))))

    texts = eval_ds["sentence"]
    labels = eval_ds["label"]

    all_preds = []
    latencies = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits
        latency_ms = (time.perf_counter() - t0) * 1000 / len(batch_texts)
        latencies.append(latency_ms)
        all_preds.extend(torch.argmax(logits, dim=-1).tolist())

    latencies_arr = np.array(latencies)
    metrics = {
        "eval_accuracy": accuracy_score(labels, all_preds),
        "eval_f1": f1_score(labels, all_preds, average="weighted"),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
        "n_samples": len(texts),
    }

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nClassification Report:")
    print(classification_report(labels, all_preds, target_names=["negative", "positive"]))

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "eval_accuracy_final": metrics["eval_accuracy"],
                "eval_f1_final": metrics["eval_f1"],
                "latency_p50_ms": metrics["latency_p50_ms"],
                "latency_p99_ms": metrics["latency_p99_ms"],
            })

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--model_uri", default=None)
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()
    evaluate(
        run_id=args.run_id,
        model_uri=args.model_uri,
        eval_split=args.eval_split,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
