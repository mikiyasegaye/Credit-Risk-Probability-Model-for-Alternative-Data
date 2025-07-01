"""
Tests for RFM analysis and target variable creation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.rfm_analysis import RFMAnalyzer, create_target_variable


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    np.random.seed(42)

    # Create dates
    dates = []
    customer_ids = []
    amounts = []

    # Create transactions with varying frequencies
    for customer_id in range(1, 11):
        # Number of transactions varies by customer (5-15)
        n_transactions = np.random.randint(5, 16)

        # Random dates within 100 days
        customer_dates = sorted(np.random.choice(100, n_transactions, replace=False))
        dates.extend(
            [datetime(2024, 1, 1) + timedelta(days=int(d)) for d in customer_dates]
        )

        # Customer IDs
        customer_ids.extend([customer_id] * n_transactions)

        # Transaction amounts based on customer segment
        if customer_id <= 3:
            # High value customers
            amounts.extend(np.random.normal(1000, 100, n_transactions))
        elif customer_id <= 7:
            # Medium value customers
            amounts.extend(np.random.normal(500, 50, n_transactions))
        else:
            # Low value customers
            amounts.extend(np.random.normal(100, 20, n_transactions))

    # Create DataFrame
    data = {
        "TransactionStartTime": dates,
        "CustomerId": customer_ids,
        "Amount": amounts,
    }

    return pd.DataFrame(data)


def test_rfm_calculation(sample_data):
    """Test RFM metrics calculation"""
    analyzer = RFMAnalyzer()
    rfm_data = analyzer.calculate_rfm(sample_data)

    assert isinstance(rfm_data, pd.DataFrame)
    assert all(col in rfm_data.columns for col in ["Recency", "Frequency", "Monetary"])
    assert len(rfm_data) == 10  # Should have one row per customer

    # Check metrics make sense
    assert all(rfm_data["Frequency"] >= 5)  # Each customer has at least 5 transactions
    assert all(rfm_data["Frequency"] <= 15)  # Each customer has at most 15 transactions
    assert all(rfm_data["Recency"] >= 0)  # Recency should be non-negative

    # Check monetary values
    high_value_customers = rfm_data.iloc[0:3]["Monetary"]
    low_value_customers = rfm_data.iloc[7:10]["Monetary"]
    assert high_value_customers.mean() > low_value_customers.mean()


def test_preprocessing(sample_data):
    """Test RFM data preprocessing"""
    analyzer = RFMAnalyzer()
    rfm_data = analyzer.calculate_rfm(sample_data)
    scaled_features = analyzer.preprocess_rfm(rfm_data)

    assert isinstance(scaled_features, np.ndarray)
    assert scaled_features.shape == (10, 3)  # 10 customers, 3 features

    # Check basic scaling properties
    means = scaled_features.mean(axis=0)
    stds = scaled_features.std(axis=0)

    # Print debug information
    print("\nScaled features:")
    print(scaled_features)
    print("\nMeans:", means)
    print("Standard deviations:", stds)

    # Means should be approximately zero
    assert np.allclose(means, 0, atol=1e-7), f"Means not close to zero: {means}"

    # Standard deviations should be approximately one
    assert np.allclose(
        stds, 1, atol=1e-7
    ), f"Standard deviations not close to one: {stds}"


def test_high_risk_identification(sample_data):
    """Test high-risk cluster identification"""
    analyzer = RFMAnalyzer()
    risk_labels = analyzer.fit_predict(sample_data)

    assert isinstance(risk_labels, pd.DataFrame)
    assert "is_high_risk" in risk_labels.columns
    assert "cluster" in risk_labels.columns
    assert set(risk_labels["is_high_risk"].unique()) == {0, 1}
    assert len(risk_labels["cluster"].unique()) == 3


def test_create_target_variable(sample_data, tmpdir):
    """Test end-to-end target variable creation"""
    risk_labels = create_target_variable(sample_data, str(tmpdir))

    # Check outputs
    assert isinstance(risk_labels, pd.DataFrame)
    assert "is_high_risk" in risk_labels.columns
    assert risk_labels["is_high_risk"].dtype == np.int64

    # Check files were created
    assert (tmpdir / "cluster_profiles.csv").exists()
    assert (tmpdir / "risk_labels.csv").exists()

    # Load and check cluster profiles
    profiles = pd.read_csv(str(tmpdir / "cluster_profiles.csv"), index_col=0)
    assert len(profiles.columns) == 3  # Should have 3 clusters
    assert all(col in profiles.index for col in ["Recency", "Frequency", "Monetary"])


def test_reproducibility():
    """Test that results are reproducible with same random_state"""
    np.random.seed(42)
    data1 = pd.DataFrame(
        {
            "TransactionStartTime": pd.date_range("2024-01-01", periods=100),
            "CustomerId": np.random.randint(1, 11, 100),
            "Amount": np.random.normal(500, 100, 100),
        }
    )

    np.random.seed(42)
    data2 = pd.DataFrame(
        {
            "TransactionStartTime": pd.date_range("2024-01-01", periods=100),
            "CustomerId": np.random.randint(1, 11, 100),
            "Amount": np.random.normal(500, 100, 100),
        }
    )

    analyzer1 = RFMAnalyzer(random_state=42)
    analyzer2 = RFMAnalyzer(random_state=42)

    labels1 = analyzer1.fit_predict(data1)
    labels2 = analyzer2.fit_predict(data2)

    pd.testing.assert_frame_equal(labels1, labels2)


if __name__ == "__main__":
    pytest.main([__file__])
