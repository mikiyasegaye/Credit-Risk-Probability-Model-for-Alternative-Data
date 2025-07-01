"""
Model training module with MLflow tracking.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn


def load_data(data_path):
    """
    Load and prepare data for training

    Parameters:
    -----------
    data_path : str
        Path to the processed features file

    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    """
    data = pd.read_csv(data_path)
    y = data["is_high_risk"]
    X = data.drop(["is_high_risk", "Unnamed: 0"], axis=1, errors="ignore")
    return X, y


def evaluate_model(model, X, y, model_name):
    """
    Evaluate model performance and log metrics to MLflow

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : array-like
        Feature matrix
    y : array-like
        True labels
    model_name : str
        Name of the model for logging
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    mlflow.log_metrics(metrics)
    print(f"\n{model_name} Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def train_model(X_train, y_train, model_type, param_grid):
    """
    Train a model with hyperparameter tuning.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    model_type : str
        Type of model to train ('logistic', 'decision_tree', or 'random_forest')
    param_grid : dict
        Grid of hyperparameters to search

    Returns:
    --------
    sklearn estimator
        The best trained model
    """
    # Initialize base model
    if model_type == "logistic":
        base_model = LogisticRegression()
    elif model_type == "decision_tree":
        base_model = DecisionTreeClassifier()
    elif model_type == "random_forest":
        base_model = RandomForestClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Perform randomized search
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=10,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_


def main():
    # Set up MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-risk-model")

    # Load data
    data_path = "data/processed/features.csv"
    X, y = load_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weights to handle imbalance
    class_weights = dict(zip(
        [0, 1],
        [1, y_train.value_counts()[0] / y_train.value_counts()[1]
         ]  # Weight high risk class more
    ))

    # Model configurations with class weights
    models = {
        "logistic": {
            "model_type": "logistic",
            "param_grid": {
                "C": np.logspace(-4, 4, 20),
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000],
                "class_weight": [class_weights],
            },
        },
        "decision_tree": {
            "model_type": "decision_tree",
            "param_grid": {
                "max_depth": [3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [class_weights],
            },
        },
        "random_forest": {
            "model_type": "random_forest",
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [class_weights],
            },
        },
    }

    # Train and evaluate each model
    best_models = {}
    for model_name, config in models.items():
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log data info
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_weights", class_weights)
            mlflow.log_param("model_type", model_name)

            # Create and train model
            model = train_model(
                X_train_scaled,
                y_train,
                config["model_type"],
                config["param_grid"],
            )

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix for {model_name}:")
            print(cm)

            # Log model
            mlflow.sklearn.log_model(
                model,
                f"{model_name}_model",
                registered_model_name="credit_risk_model"
            )

            best_models[model_name] = {
                "model": model,
                "metrics": metrics,
            }

            print(f"\nMetrics for {model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

    # Compare models and select the best one
    best_model_name = max(
        best_models.keys(),
        key=lambda k: best_models[k]["metrics"]["f1"]
    )
    print(f"\nBest model: {best_model_name}")
    print("Best model metrics:")
    for metric_name, value in best_models[best_model_name]["metrics"].items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
