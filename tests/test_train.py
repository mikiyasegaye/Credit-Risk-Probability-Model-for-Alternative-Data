"""
Unit tests for model training functions.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.train import load_data, train_model, evaluate_model


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100

    # Create synthetic features
    X = np.random.randn(n_samples, 3)
    # Create synthetic target (binary classification)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

    return pd.DataFrame(X, columns=["feature1", "feature2", "feature3"]), pd.Series(y)


@pytest.fixture
def temp_data_file(sample_data, tmp_path):
    """Save sample data to temporary file"""
    file_path = tmp_path / "test_features.csv"
    pd.concat(
        [sample_data[0], pd.Series(sample_data[1], name="is_high_risk")], axis=1
    ).to_csv(file_path)
    return file_path


def test_load_data(temp_data_file):
    """Test data loading function"""
    X, y = load_data(temp_data_file)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 3  # Three features
    assert len(y) == len(X)
    assert y.name == "is_high_risk"
    assert set(y.unique()) == {0, 1}  # Binary classification


def test_train_model_logistic(sample_data):
    """Test logistic regression model training"""
    X, y = sample_data

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = train_model(X_scaled, y, model_type="logistic")

    # Check model type
    assert str(type(model).__name__) == "LogisticRegression"

    # Check predictions
    predictions = model.predict(X_scaled)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    assert set(predictions).issubset({0, 1})


def test_train_model_decision_tree(sample_data):
    """Test decision tree model training"""
    X, y = sample_data

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model with specific parameters
    params = {"max_depth": 3, "min_samples_split": 2}
    model = train_model(X_scaled, y, model_type="decision_tree", params=params)

    # Check model type
    assert str(type(model).__name__) == "DecisionTreeClassifier"

    # Check if parameters were set correctly
    assert model.max_depth == params["max_depth"]
    assert model.min_samples_split == params["min_samples_split"]

    # Check predictions
    predictions = model.predict(X_scaled)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    assert set(predictions).issubset({0, 1})


def test_train_model_random_forest():
    """Test random forest model training"""
    X = np.random.normal(0, 1, (100, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = train_model(X, y, model_type="random_forest")
    assert isinstance(model, RandomForestClassifier)

    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert set(y_pred).issubset({0, 1})


def test_train_model_invalid_type(sample_data):
    """Test error handling for invalid model type"""
    X, y = sample_data

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try to train with invalid model type
    with pytest.raises(ValueError) as exc_info:
        train_model(X_scaled, y, model_type="invalid_model")

    assert "Unknown model type" in str(exc_info.value)


def test_train_model_with_params():
    """Test model training with custom parameters"""
    X = np.random.normal(0, 1, (100, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    params = {"max_depth": 3, "min_samples_split": 5}
    model = train_model(X, y, model_type="decision_tree", params=params)

    assert model.max_depth == 3
    assert model.min_samples_split == 5


def test_evaluate_model(sample_data):
    """Test model evaluation function"""
    X, y = sample_data

    # Scale features and split data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a simple model
    model = train_model(X_scaled, y, model_type="logistic")

    # Capture printed output
    import io
    import sys

    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Evaluate model
    evaluate_model(model, X_scaled, y, "test_model")

    # Restore stdout
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Check if all required metrics are present
    required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in required_metrics:
        assert metric in output.lower()
