import mlflow
import pandas as pd


def load_model(run_id, model_name):
    """
    Load a model from MLflow tracking.

    Parameters:
    -----------
    run_id : str
        The run ID from MLflow
    model_name : str
        Name of the model to load ('logistic', 'decision_tree', or 'random_forest')

    Returns:
    --------
    model : sklearn estimator
        The loaded model
    """
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict(model, data):
    """
    Make predictions using the loaded model.

    Parameters:
    -----------
    model : sklearn estimator
        The loaded model
    data : pandas.DataFrame
        Features to make predictions on

    Returns:
    --------
    predictions : numpy.ndarray
        Model predictions
    probabilities : numpy.ndarray
        Prediction probabilities for the positive class
    """
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    return predictions, probabilities


if __name__ == "__main__":
    # Example usage:
    # Replace with actual run_id from MLflow UI
    run_id = "your_run_id_here"
    model_name = "decision_tree"  # or "logistic" or "random_forest"

    # Load the model
    model = load_model(run_id, model_name)

    # Load data for prediction
    data = pd.read_csv("path_to_your_features.csv")

    # Make predictions
    predictions, probabilities = predict(model, data)
    print("Predictions:", predictions[:5])
    print("Probabilities:", probabilities[:5])
