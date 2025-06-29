import mlflow
from mlflow.tracking import MlflowClient
import os


def register_best_model():
    # Set up MLflow tracking URI with SQLite backend
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    print(f"Using MLflow tracking URI: {tracking_uri}")

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
    best_run = runs.sort_values('metrics.roc_auc', ascending=False).iloc[0]
    best_run_id = best_run.run_id
    roc_auc = best_run['metrics.roc_auc']
    model_type = best_run['params.model_type']

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
        print(f"\nRegistering model as '{registered_model_name}'...")

        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(registered_model_name)
            print(f"Created new registered model: {registered_model_name}")
        except Exception as e:
            if "already exists" not in str(e):
                raise e
            print(f"Model {registered_model_name} already exists")

        # Create new version
        version = client.create_model_version(
            name=registered_model_name,
            source=model_path,
            run_id=best_run_id
        )
        print(
            f"Created version {version.version} for model {registered_model_name}")

        # Transition the model to Production stage
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version.version,
            stage="Production"
        )
        print(f"Transitioned version {version.version} to Production stage")

        print("\nModel Registration Complete!")
        print("\nTo view and compare models in the MLflow UI:")
        print("1. Go to http://localhost:5001")
        print("2. Click on 'Models' in the left sidebar to see registered models")
        print("3. Click on 'Experiments' to compare different runs")
        print("\nRegistered Model Details:")
        print(f"Name: {registered_model_name}")
        print(f"Version: {version.version}")
        print(f"Stage: Production")
        print(f"Source Run: {best_run_id}")
        print(f"Model Type: {model_type}")
        print(f"ROC-AUC: {roc_auc:.4f}")

    except Exception as e:
        print(f"Error registering model: {str(e)}")
        raise e


if __name__ == "__main__":
    register_best_model()
