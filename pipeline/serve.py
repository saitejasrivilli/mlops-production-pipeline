"""
BentoML service for the MLflow text-classifier model.

To build and serve:
    python pipeline/serve.py --save          # save bento model
    bentoml serve pipeline/serve.py:svc      # serve locally on :3000

Endpoints:
    POST /classify        {"text": "..."} -> {"label": str, "confidence": float}
    POST /batch_classify  [{"text": "..."}, ...] -> [{"label", "confidence"}, ...]
    GET  /health          -> {"status": "ok"}
"""
import argparse
import numpy as np
import torch
import mlflow.pytorch
import bentoml
from bentoml.io import JSON, Text
from transformers import AutoTokenizer

MODEL_TAG = "text-classifier"
TOKENIZER_NAME = "distilbert-base-uncased"
LABELS = ["negative", "positive"]


def load_from_mlflow(stage: str = "Staging") -> tuple:
    model_uri = f"models:/{MODEL_TAG}/{stage}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return model, tokenizer


def save_to_bentoml(stage: str = "Staging") -> str:
    model, tokenizer = load_from_mlflow(stage)
    saved = bentoml.pytorch.save_model(
        MODEL_TAG,
        model,
        custom_objects={"tokenizer": tokenizer},
        signatures={"__call__": {"batchable": True}},
        metadata={"stage": stage, "labels": LABELS},
    )
    print(f"Saved BentoML model: {saved.tag}")
    return str(saved.tag)


runner = bentoml.pytorch.get(f"{MODEL_TAG}:latest").to_runner()

svc = bentoml.Service(MODEL_TAG, runners=[runner])


@svc.api(input=JSON(), output=JSON())
async def classify(payload: dict) -> dict:
    text = payload.get("text", "")
    tokenizer = runner.custom_objects["tokenizer"]
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = await runner.async_run(**inputs)
    probs = torch.softmax(logits, dim=-1)[0]
    label_idx = int(torch.argmax(probs))
    return {
        "label": LABELS[label_idx],
        "confidence": float(probs[label_idx]),
        "scores": {l: float(p) for l, p in zip(LABELS, probs)},
    }


@svc.api(input=JSON(), output=JSON())
async def batch_classify(payload: list) -> list:
    texts = [item.get("text", "") for item in payload]
    tokenizer = runner.custom_objects["tokenizer"]
    inputs = tokenizer(
        texts, truncation=True, max_length=128, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = await runner.async_run(**inputs)
    probs = torch.softmax(logits, dim=-1)
    results = []
    for i, p in enumerate(probs):
        label_idx = int(torch.argmax(p))
        results.append({
            "text": texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i],
            "label": LABELS[label_idx],
            "confidence": float(p[label_idx]),
        })
    return results


@svc.api(input=Text(), output=JSON())
async def health(_: str) -> dict:
    return {"status": "ok", "model": MODEL_TAG}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save MLflow model to BentoML")
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()
    if args.save:
        tag = save_to_bentoml(args.stage)
        print(f"Run: bentoml serve pipeline/serve.py:svc --reload")
