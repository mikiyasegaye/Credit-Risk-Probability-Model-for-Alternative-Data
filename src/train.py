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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
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
    y = data['is_high_risk']
    X = data.drop(['is_high_risk', 'Unnamed: 0'], axis=1, errors='ignore')
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
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_prob)
    }

    mlflow.log_metrics(metrics)
    print(f"\n{model_name} Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def train_model(X_train, y_train, model_type='logistic', params=None):
    """
    Train a model with the specified parameters

    Parameters:
    -----------
    X_train : array-like
        Training feature matrix
    y_train : array-like
        Training labels
    model_type : str
        Type of model to train ('logistic', 'decision_tree', 'random_forest')
    params : dict
        Model hyperparameters

    Returns:
    --------
    model : sklearn estimator
        Trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(**(params or {}))
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(**(params or {}))
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**(params or {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def main():
    # Set up MLflow tracking with SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-model")

    # Load data
    data_path = "data/processed/features.csv"
    X, y = load_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model configurations
    models = {
        'logistic': {
            'model_type': 'logistic',
            'param_grid': {
                'C': np.logspace(-4, 4, 20),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
        },
        'decision_tree': {
            'model_type': 'decision_tree',
            'param_grid': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'random_forest': {
            'model_type': 'random_forest',
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }

    # Train and evaluate models
    for model_name, config in models.items():
        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):
            # Log model type
            mlflow.log_param('model_type', model_name)

            # Train model with hyperparameter tuning
            search = RandomizedSearchCV(
                train_model(X_train_scaled, y_train,
                            model_type=config['model_type']),
                config['param_grid'],
                n_iter=10,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            search.fit(X_train_scaled, y_train)

            # Log best parameters
            for param, value in search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)

            # Evaluate on train set
            print("\nTrain Set Performance:")
            evaluate_model(search.best_estimator_,
                           X_train_scaled, y_train, model_name)

            # Evaluate on test set
            print("\nTest Set Performance:")
            evaluate_model(search.best_estimator_,
                           X_test_scaled, y_test, model_name)

            # Save model
            mlflow.sklearn.log_model(search.best_estimator_, model_name)


if __name__ == "__main__":
    main()
