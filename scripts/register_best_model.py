import mlflow
from mlflow.tracking import MlflowClient
import os
import time
import requests
from urllib.parse import urlparse


def wait_for_mlflow_server(tracking_uri, max_retries=10, delay=5):
    parsed_uri = urlparse(tracking_uri)
    if parsed_uri.scheme == "http":
        health_url = tracking_uri.rstrip("/") + "/health"
        for i in range(max_retries):
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    print("MLflow server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            print(
                f"Waiting for MLflow server (attempt {i+1}/{max_retries})...")
            time.sleep(delay)
        raise Exception("MLflow server not available after maximum retries")
    return True


def register_best_model():
    # Set up MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    print(f"Using MLflow tracking URI: {tracking_uri}")

    # Wait for MLflow server to be ready
    wait_for_mlflow_server(tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Search for the experiment
    experiment = mlflow.get_experiment_by_name("credit-risk-model")
    if experiment is None:
        print("No experiment found with name 'credit-risk-model'")
        return

    print(f"Found experiment with ID: {experiment.experiment_id}")

    # Get all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        print("No runs found in the experiment")
        return

    print(f"Found {len(runs)} runs")

    # Find the best run based on ROC-AUC score
    best_run = runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]
    best_run_id = best_run.run_id
    roc_auc = best_run["metrics.roc_auc"]
    model_type = best_run["params.model_type"] + "_model"

    print(f"\nBest performing model:")
    print(f"Model Type: {model_type}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Run ID: {best_run_id}")

    # Get the model artifact path
    run = mlflow.get_run(best_run_id)
    artifact_uri = run.info.artifact_uri
    model_path = os.path.join(artifact_uri, model_type)

    print(f"\nModel artifact path: {model_path}")

    # Register the model
    try:
        registered_model_name = "credit_risk_model"
        print(
            f"\nChecking existing model versions for '{registered_model_name}'...")

        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(registered_model_name)
            print(f"Created new registered model: {registered_model_name}")
        except Exception as e:
            if "already exists" not in str(e):
                raise e
            print(f"Model {registered_model_name} already exists")

        # Check existing versions
        versions = client.search_model_versions(
            f"name='{registered_model_name}'")
        production_versions = [
            v for v in versions if v.current_stage == "Production"]

        # Check if we already have this run registered
        existing_version = None
        for version in versions:
            if version.run_id == best_run_id:
                existing_version = version
                break

        if existing_version:
            print(
                f"\nThis run is already registered as version {existing_version.version}")
            if existing_version.current_stage != "Production":
                # Archive current production version if it exists
                if production_versions:
                    for pv in production_versions:
                        client.transition_model_version_stage(
                            name=registered_model_name,
                            version=pv.version,
                            stage="Archived"
                        )
                        print(f"Archived version {pv.version}")

                # Transition this version to Production
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=existing_version.version,
                    stage="Production"
                )
                print(
                    f"Transitioned version {existing_version.version} to Production stage")
        else:
            # Archive current production version if it exists
            if production_versions:
                for pv in production_versions:
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=pv.version,
                        stage="Archived"
                    )
                    print(f"Archived version {pv.version}")

            # Create new version
            version = client.create_model_version(
                name=registered_model_name,
                source=model_path,
                run_id=best_run_id,
                tags={"artifact_path": model_type}
            )
            print(
                f"Created version {version.version} for model {registered_model_name}")

            # Transition the model to Production stage
            client.transition_model_version_stage(
                name=registered_model_name,
                version=version.version,
                stage="Production"
            )
            print(
                f"Transitioned version {version.version} to Production stage")

        print("\nModel Registration Complete!")
        print("\nTo view and compare models in the MLflow UI:")
        print(f"1. Go to {tracking_uri}")
        print("2. Click on 'Models' in the left sidebar to see registered models")
        print("3. Click on 'Experiments' to compare different runs")
        print("\nRegistered Model Details:")
        print(f"Name: {registered_model_name}")
        if existing_version:
            print(f"Version: {existing_version.version} (already registered)")
        else:
            print(f"Version: {version.version} (newly registered)")
        print(f"Stage: Production")
        print(f"Source Run: {best_run_id}")
        print(f"Model Type: {model_type}")
        print(f"ROC-AUC: {roc_auc:.4f}")

    except Exception as e:
        print(f"Error registering model: {str(e)}")
        raise e


if __name__ == "__main__":
    register_best_model()
