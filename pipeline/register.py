import argparse
import sys
import mlflow
from mlflow.tracking import MlflowClient


def register_if_better(
    run_id: str = None,
    metric: str = "eval_accuracy",
    threshold: float = 0.90,
    model_name: str = "text-classifier",
    stage: str = "Staging",
) -> bool:
    if run_id is None:
        try:
            with open("last_run_id.txt") as f:
                run_id = f.read().strip()
        except FileNotFoundError:
            raise ValueError("Provide --run_id or run train.py first.")

    client = MlflowClient()
    run = client.get_run(run_id)
    metric_value = run.data.metrics.get(metric) or run.data.metrics.get(f"{metric}_final")

    if metric_value is None:
        print(f"Metric '{metric}' not found in run {run_id}. Available: {list(run.data.metrics.keys())}")
        return False

    print(f"Run {run_id[:8]}: {metric} = {metric_value:.4f} (threshold = {threshold})")

    if metric_value < threshold:
        print(f"SKIP: {metric_value:.4f} < {threshold}. Model not registered.")
        return False

    # Check if there's already a model in the target stage to compare against
    try:
        current_versions = client.get_latest_versions(model_name, stages=[stage])
        if current_versions:
            current_run_id = current_versions[0].run_id
            current_run = client.get_run(current_run_id)
            current_metric = current_run.data.metrics.get(metric) or current_run.data.metrics.get(f"{metric}_final", 0.0)
            if metric_value <= current_metric:
                print(f"SKIP: New model ({metric_value:.4f}) <= current {stage} model ({current_metric:.4f}).")
                return False
            print(f"Replacing current {stage} model ({current_metric:.4f}) with new ({metric_value:.4f}).")
    except Exception:
        pass

    # Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )

    client.set_model_version_tag(model_name, version, "promoted_by", "ci_pipeline")
    client.set_model_version_tag(model_name, version, "eval_accuracy", str(metric_value))

    print(f"Model '{model_name}' v{version} promoted to {stage}. Accuracy: {metric_value:.4f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--metric", default="eval_accuracy")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--model_name", default="text-classifier")
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()
    success = register_if_better(
        run_id=args.run_id,
        metric=args.metric,
        threshold=args.threshold,
        model_name=args.model_name,
        stage=args.stage,
    )
    sys.exit(0 if success else 1)
