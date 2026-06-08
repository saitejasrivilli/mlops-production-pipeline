import argparse
import os
import time
import mlflow
import mlflow.pytorch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train(
    model_name: str = "distilbert-base-uncased",
    dataset_name: str = "sst2",
    run_name: str = "baseline",
    output_dir: str = "models/",
    num_epochs: int = 1,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_samples: int = 1000,
) -> str:
    mlflow.set_experiment("ml-pipeline-demo")

    ds = load_dataset("glue", dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    train_ds = ds["train"].select(range(min(max_samples, len(ds["train"]))))
    val_ds = ds["validation"].select(range(min(200, len(ds["validation"]))))
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model_name": model_name,
            "dataset": dataset_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_samples": max_samples,
        })

        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0

        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "eval_accuracy": eval_results["eval_accuracy"],
            "eval_f1": eval_results["eval_f1"],
            "eval_loss": eval_results["eval_loss"],
            "train_time_seconds": train_time,
        })

        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="text-classifier",
        )

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
        with open("last_run_id.txt", "w") as f:
            f.write(run_id)
        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--output_dir", default="models/")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()
    train(
        model_name=args.model_name,
        dataset_name=args.dataset,
        run_name=args.run_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
