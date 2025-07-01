"""
Tests for the data processing pipeline
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from src.data_processing import (
    DateFeatureExtractor,
    CustomerAggregator,
    CategoryEncoder,
    process_and_save_data,
    FEATURE_GROUPS,
    preprocess_features,
    handle_missing_values,
    encode_categorical_features,
    load_data,
    evaluate_model,
)


@pytest.fixture
def temp_output_dir(tmpdir):
    """Create a temporary directory for test outputs"""
    return str(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)

    # Create sample dates
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]

    # Create sample data
    data = {
        "TransactionStartTime": dates,
        "CustomerId": np.random.randint(1, 11, 100),  # 10 unique customers
        "Amount": np.random.normal(1000, 200, 100),
        # Absolute value for transaction value
        "Value": np.abs(np.random.normal(1000, 200, 100)),
        "ProductCategory": np.random.choice(["A", "B", "C"], 100),
        "PaymentMethod": np.random.choice(["Credit", "Debit", "Cash"], 100),
        "Status": np.random.choice(["Success", "Failed"], 100),
        "ChannelId": np.random.choice(["Web", "Android", "iOS"], 100),
        "CurrencyCode": np.random.choice(["USD", "EUR", "GBP"], 100),
        "CountryCode": np.random.choice(["US", "UK", "FR"], 100),
        "ProviderId": np.random.choice(["P1", "P2", "P3"], 100),
        "ProductId": np.random.choice(["Prod1", "Prod2", "Prod3"], 100),
        "PricingStrategy": np.random.choice(["Standard", "Premium", "Basic"], 100),
        # 5 unique subscriptions
        "SubscriptionId": np.random.randint(1, 6, 100),
        "TransactionId": range(1, 101),
        "BatchId": np.random.randint(1, 21, 100),
        "AccountId": np.random.randint(1, 11, 100),
        # Binary target for testing
        "FraudResult": np.random.randint(0, 2, 100),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing"""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "numeric_feature": np.random.normal(0, 1, n_samples),
            "categorical_feature": np.random.choice(["A", "B", "C", None], n_samples),
            "binary_feature": np.random.choice([0, 1], n_samples),
            "missing_feature": np.random.choice([np.nan, 1, 2, 3], n_samples),
        }
    )

    return data


def test_date_feature_extractor():
    """Test the DateFeatureExtractor transformer"""
    # Create sample data
    data = pd.DataFrame(
        {
            "TransactionStartTime": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "OtherColumn": [1, 2],
        }
    )

    # Apply transformation
    transformer = DateFeatureExtractor()
    result = transformer.fit_transform(data)

    # Check if date features were created
    expected_features = FEATURE_GROUPS["temporal_features"]
    assert all(col in result.columns for col in expected_features)
    assert "TransactionStartTime" not in result.columns


def test_customer_aggregator(sample_data):
    """Test the CustomerAggregator transformer"""
    # Apply transformation
    transformer = CustomerAggregator()
    result = transformer.fit_transform(sample_data)

    # Check if aggregated features were created
    expected_features = FEATURE_GROUPS["customer_aggregations"]
    assert all(col in result.columns for col in expected_features)

    # Check if aggregations are correct for a specific customer
    customer_id = sample_data["CustomerId"].iloc[0]
    customer_transactions = sample_data[sample_data["CustomerId"] == customer_id]

    assert result[result["CustomerId"] == customer_id]["Transaction_Count"].iloc[
        0
    ] == len(customer_transactions)
    assert np.isclose(
        result[result["CustomerId"] == customer_id]["Total_Transaction_Amount"].iloc[0],
        customer_transactions["Amount"].sum(),
    )
    assert result[result["CustomerId"] == customer_id][
        "Unique_Product_Categories"
    ].iloc[0] == len(customer_transactions["ProductCategory"].unique())


def test_category_encoder_onehot(sample_data):
    """Test the CategoryEncoder with one-hot encoding"""
    # Apply transformation
    categorical_cols = FEATURE_GROUPS["categorical_features"]
    transformer = CategoryEncoder(categorical_columns=categorical_cols, method="onehot")
    result = transformer.fit_transform(sample_data)

    # Check if categorical columns were encoded
    for col in categorical_cols:
        # Original column should be dropped
        assert col not in result.columns
        # Check if at least one encoded column exists for each category
        assert any(c.startswith(col + "_") for c in result.columns)


def test_category_encoder_label(sample_data):
    """Test the CategoryEncoder with label encoding"""
    # Apply transformation
    categorical_cols = FEATURE_GROUPS["categorical_features"]
    transformer = CategoryEncoder(categorical_columns=categorical_cols, method="label")
    result = transformer.fit_transform(sample_data)

    # Check if categorical columns were encoded
    for col in categorical_cols:
        # Column should still exist
        assert col in result.columns
        # Values should be numeric
        assert pd.api.types.is_numeric_dtype(result[col])


def test_process_and_save_data(sample_data, temp_output_dir):
    """Test the complete data processing pipeline and file outputs"""
    # Process data with different encoding methods
    for method in ["onehot", "label"]:
        # Process the data with target variable
        target = sample_data["FraudResult"]
        result = process_and_save_data(
            sample_data, temp_output_dir, target=target, categorical_method=method
        )

        # Check if result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check if output files were created
        assert os.path.exists(os.path.join(temp_output_dir, "features.csv"))
        assert os.path.exists(os.path.join(temp_output_dir, "raw_for_rfm.csv"))
        assert os.path.exists(os.path.join(temp_output_dir, "metadata.json"))

        # Check raw_for_rfm.csv content
        rfm_data = pd.read_csv(os.path.join(temp_output_dir, "raw_for_rfm.csv"))
        expected_columns = (
            FEATURE_GROUPS["id_columns"] + FEATURE_GROUPS["rfm_base_features"]
        )
        assert all(col in rfm_data.columns for col in expected_columns)

        # Check metadata.json content
        with open(os.path.join(temp_output_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            assert "feature_groups" in metadata
            assert "transformers" in metadata
            assert "creation_timestamp" in metadata
            assert "pipeline_steps" in metadata


def test_feature_groups():
    """Test the feature groups definition"""
    # Check if all required feature groups exist
    required_groups = [
        "id_columns",
        "rfm_base_features",
        "categorical_features",
        "temporal_features",
        "customer_aggregations",
    ]
    assert all(group in FEATURE_GROUPS for group in required_groups)

    # Check if there are no duplicate features across groups
    all_features = []
    for group in FEATURE_GROUPS.values():
        all_features.extend(group)
    assert len(all_features) == len(
        set(all_features)
    ), "Duplicate features found across groups"


def test_load_data(tmp_path):
    """Test the load_data function"""
    # Create a temporary test dataset
    test_data = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "is_high_risk": [0, 1, 0]}
    )

    # Save test data without index
    test_file = tmp_path / "test_features.csv"
    test_data.to_csv(test_file, index=False)

    # Load data using the function
    X, y = load_data(test_file)

    # Check that features and target are correctly separated
    assert "is_high_risk" not in X.columns
    assert isinstance(y, pd.Series)
    assert y.name == "is_high_risk"

    # Check that unnamed columns are removed
    assert not any("Unnamed" in col for col in X.columns)

    # Check shapes
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 2  # Only feature1 and feature2


def test_evaluate_model():
    """Test the evaluate_model function"""
    # Create test data
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_test = np.array([0, 1, 0, 1])

    # Create and fit a simple model
    model = LogisticRegression()
    model.fit(X_test, y_test)

    # Get evaluation metrics
    metrics = evaluate_model(model, X_test, y_test, model_name="test_model")

    # Check that all required metrics are present
    required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    assert all(metric in metrics for metric in required_metrics)

    # Check that metric values are in valid ranges
    for metric, value in metrics.items():
        assert 0 <= value <= 1, f"{metric} should be between 0 and 1"


def test_handle_missing_values(sample_raw_data):
    """Test handling of missing values"""
    data = sample_raw_data.copy()

    # Count initial missing values
    initial_missing = data.isnull().sum().sum()
    assert initial_missing > 0, "Test data should contain missing values"

    # Handle missing values
    cleaned_data = handle_missing_values(data)

    # Check that there are no missing values after cleaning
    final_missing = cleaned_data.isnull().sum().sum()
    assert final_missing == 0, "All missing values should be handled"

    # Check that the number of rows remains the same
    assert len(cleaned_data) == len(data), "Number of rows should not change"


def test_encode_categorical_features(sample_raw_data):
    """Test categorical feature encoding"""
    data = sample_raw_data.copy()

    # Handle missing values first
    data = handle_missing_values(data)

    # Get initial categorical columns
    categorical_cols = data.select_dtypes(include=["object"]).columns
    assert len(categorical_cols) > 0, "Test data should contain categorical features"

    # Encode categorical features
    encoded_data = encode_categorical_features(data)

    # Check that there are no object dtype columns left
    remaining_categorical = encoded_data.select_dtypes(include=["object"]).columns
    assert len(remaining_categorical) == 0, "All categorical features should be encoded"

    # Check that encoding created dummy variables for each category
    for col in categorical_cols:
        unique_values = data[col].nunique()
        dummy_cols = [c for c in encoded_data.columns if c.startswith(f"{col}_")]
        assert (
            len(dummy_cols) == unique_values
        ), f"Wrong number of dummy variables for {col}"


def test_preprocess_features(sample_raw_data):
    """Test complete preprocessing pipeline"""
    data = sample_raw_data.copy()

    # Process features
    processed_data = preprocess_features(data)

    # Check that output is a pandas DataFrame
    assert isinstance(
        processed_data, pd.DataFrame
    ), "Output should be a pandas DataFrame"

    # Check that there are no missing values
    assert (
        processed_data.isnull().sum().sum() == 0
    ), "Processed data should not contain missing values"

    # Check that there are no categorical features
    assert (
        len(processed_data.select_dtypes(include=["object"]).columns) == 0
    ), "Processed data should not contain categorical features"

    # Check that all features are numeric
    assert all(processed_data.dtypes != "object"), "All features should be numeric"


def test_preprocess_features_empty_data():
    """Test preprocessing with empty DataFrame"""
    empty_data = pd.DataFrame()

    with pytest.raises(ValueError) as exc_info:
        preprocess_features(empty_data)

    assert "Empty DataFrame" in str(exc_info.value)


def test_preprocess_features_single_value():
    """Test preprocessing with single value columns"""
    data = pd.DataFrame({"constant": [1] * 100, "normal": np.random.normal(0, 1, 100)})

    processed_data = preprocess_features(data)

    # Check that constant column is dropped
    assert (
        "constant" not in processed_data.columns
    ), "Constant value columns should be dropped"
    assert "normal" in processed_data.columns, "Non-constant columns should be retained"


if __name__ == "__main__":
    pytest.main([__file__])
