import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import pytest
from src.api.main import preprocess_features
import os


def create_sample_data():
    """Create a small sample dataset for testing"""
    # Create sample data with RFM metrics and derived features
    sample_data = {
        "Recency": [5, 30, 100],  # Days since last transaction
        "Frequency": [50, 10, 2],  # Number of transactions
        "Monetary": [10000, 1000, 100],  # Total amount spent
        # Log-transformed frequency
        "Frequency_Log": [np.log1p(50), np.log1p(10), np.log1p(2)],
        # Log-transformed monetary
        "Monetary_Log": [np.log1p(10000), np.log1p(1000), np.log1p(100)],
        "avg_transaction_value": [200, 100, 50],  # Average transaction value
        "transaction_frequency": [10, 5, 1],  # Transactions per month
        "spending_variance": [500, 200, 50],  # Variance in spending
        "max_transaction": [1000, 500, 200],  # Maximum transaction amount
        "min_transaction": [50, 20, 10],  # Minimum transaction amount
        # Total transaction amount
        "total_transactions_amount": [10000, 1000, 100],
        "transaction_consistency": [0.8, 0.5, 0.2],  # Consistency score
        "recent_transaction_trend": [1.2, 0.9, 0.5],  # Recent trend
        "seasonal_pattern": [1, 0, 0],  # Seasonal pattern detected
        # Ratio of weekend transactions
        "weekend_transactions_ratio": [0.3, 0.4, 0.6],
        # Ratio of evening transactions
        "evening_transactions_ratio": [0.2, 0.3, 0.5],
        "transaction_day_variance": [2, 5, 10],  # Variance in transaction days
        # Growth in transaction amounts
        "transaction_amount_growth": [0.1, -0.1, -0.3],
        # Average days between transactions
        "days_between_transactions": [7, 15, 30],
        "inactive_periods": [1, 3, 5],  # Number of inactive periods
        "cluster": [0, 1, 2],  # Cluster assignment
    }
    return pd.DataFrame(sample_data)


def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")

    # Create sample data
    print("\nCreating sample data...")
    sample_data = create_sample_data()
    print("\nSample data (first few columns):")
    print(
        sample_data[
            [
                "Recency",
                "Frequency",
                "Monetary",
                "avg_transaction_value",
                "transaction_frequency",
            ]
        ].to_string()
    )

    # Scale the features (important to use the same scaling as training)
    print("\nScaling features...")
    scaler = StandardScaler()
    sample_data_scaled = scaler.fit_transform(sample_data)
    sample_data_scaled = pd.DataFrame(
        sample_data_scaled, columns=sample_data.columns)

    # Load the model
    print("\nLoading model...")
    try:
        # List all experiments
        experiments = mlflow.search_experiments()
        for exp in experiments:
            print(f"\nExperiment: {exp.name}")

            # Get all runs in the experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

            if not runs.empty:
                # Find the run with the best ROC-AUC score
                best_run = runs.sort_values(
                    "metrics.roc_auc", ascending=False).iloc[0]
                run_id = best_run.run_id
                print(f"Best run ID: {run_id}")
                print(f"ROC-AUC: {best_run['metrics.roc_auc']}")

                # Load the model from this run
                model_uri = f"runs:/{run_id}/decision_tree"
                model = mlflow.sklearn.load_model(model_uri)

                # Make predictions
                print("\nMaking predictions...")
                predictions = model.predict(sample_data_scaled)
                probabilities = model.predict_proba(sample_data_scaled)[:, 1]

                # Print results
                print("\nResults:")
                for i in range(len(predictions)):
                    print(f"\nSample {i+1}:")
                    print("Key metrics:")
                    print(
                        f"  Recency: {sample_data.iloc[i]['Recency']} days since last transaction"
                    )
                    print(
                        f"  Frequency: {sample_data.iloc[i]['Frequency']} transactions"
                    )
                    print(
                        f"  Monetary: ${sample_data.iloc[i]['Monetary']:.2f} total spent"
                    )
                    print(
                        f"  Avg Transaction: ${sample_data.iloc[i]['avg_transaction_value']:.2f}"
                    )
                    print(
                        f"Prediction: {'High Risk' if predictions[i] == 1 else 'Low Risk'}"
                    )
                    print(f"Probability of High Risk: {probabilities[i]:.2%}")

                break  # We found and used the best model, so we can stop
            else:
                print("No runs found in this experiment")

    except Exception as e:
        print(f"Error: {str(e)}")


def test_preprocess_features_basic():
    """Test basic preprocessing with different risk scenarios"""
    # High risk scenario (low values)
    high_risk_input = {
        "monetary": 20.0,
        "frequency": 1,
        "avg_transaction_amount": 20.0,
        "customer_tenure": 10.0
    }

    # Medium risk scenario
    medium_risk_input = {
        "monetary": 500.0,
        "frequency": 5,
        "avg_transaction_amount": 100.0,
        "customer_tenure": 90.0
    }

    # Low risk scenario (high values)
    low_risk_input = {
        "monetary": 5000.0,
        "frequency": 20,
        "avg_transaction_amount": 250.0,
        "customer_tenure": 365.0
    }

    # Process each scenario
    high_risk_features = preprocess_features(high_risk_input)
    medium_risk_features = preprocess_features(medium_risk_input)
    low_risk_features = preprocess_features(low_risk_input)

    # Verify feature engineering is correct
    def verify_features(input_data, processed_df):
        # Verify all 21 expected features are present
        expected_features = [
            'Value', 'Amount', 'Avg_Value', 'Std_Transaction_Amount',
            'Avg_Transaction_Amount', 'Max_Transaction_Amount', 'Total_Transaction_Amount',
            'PricingStrategy_0.0', 'Min_Transaction_Amount', 'ProductId_ProductId_9',
            'Total_Value', 'Transaction_Count', 'ProviderId_ProviderId_3',
            'ProviderId_ProviderId_1', 'Is_Weekend', 'Unique_Product_Categories',
            'ProviderId_ProviderId_5', 'ProductId_ProductId_15', 'Unique_Providers',
            'ProductId_ProductId_5', 'FraudResult'
        ]
        assert all(
            feature in processed_df.columns for feature in expected_features)
        assert len(processed_df.columns) == len(expected_features)

        # Verify feature relationships are maintained after scaling
        assert processed_df['Total_Value'].iloc[0] == processed_df['Value'].iloc[0]
        assert processed_df['Amount'].iloc[0] == processed_df['Avg_Transaction_Amount'].iloc[0]
        assert processed_df['Max_Transaction_Amount'].iloc[0] == processed_df['Amount'].iloc[0]
        assert processed_df['Min_Transaction_Amount'].iloc[0] == processed_df['Amount'].iloc[0]

    verify_features(high_risk_input, high_risk_features)
    verify_features(medium_risk_input, medium_risk_features)
    verify_features(low_risk_input, low_risk_features)


def test_preprocess_features_edge_cases():
    """Test preprocessing with edge cases"""
    # Edge case: Minimum valid values
    min_input = {
        "monetary": 0.01,
        "frequency": 1,
        "avg_transaction_amount": 0.01,
        "customer_tenure": 1.0
    }

    # Edge case: Very high values
    max_input = {
        "monetary": 1000000.0,
        "frequency": 1000,
        "avg_transaction_amount": 1000.0,
        "customer_tenure": 3650.0
    }

    # Process edge cases
    min_features = preprocess_features(min_input)
    max_features = preprocess_features(max_input)

    # Verify no NaN or infinite values
    assert not min_features.isna().any().any()
    assert not max_features.isna().any().any()
    assert not np.isinf(min_features.values).any()
    assert not np.isinf(max_features.values).any()


def test_model_predictions():
    """Test actual model predictions with different risk scenarios"""
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

    # Load the production model
    model_name = "credit_risk_model"
    model = mlflow.sklearn.load_model(
        f"models:/{model_name}/4")  # Decision tree model

    # High risk scenario (very low values, matching training data)
    high_risk_input = {
        "monetary": 1.0,  # Very low total value
        "frequency": 1,    # Single transaction
        "avg_transaction_amount": 1.0,  # Very low amount
        "customer_tenure": 1.0  # New customer
    }

    # Medium risk scenario (moderate values)
    medium_risk_input = {
        "monetary": 100.0,  # Moderate total value
        "frequency": 10,     # Few transactions
        "avg_transaction_amount": 10.0,  # Low-moderate amount
        "customer_tenure": 30.0  # Month-old customer
    }

    # Low risk scenario (high values, matching training data)
    low_risk_input = {
        "monetary": 10000.0,  # High total value
        "frequency": 100,      # Many transactions
        "avg_transaction_amount": 100.0,  # High amount
        "customer_tenure": 365.0  # Year-old customer
    }

    # Process and predict each scenario
    def get_prediction(input_data):
        features = preprocess_features(input_data)
        print(f"\nFeatures for {input_data}:")
        print("Feature names:", list(features.columns))
        print("Feature values:", features.iloc[0].to_dict())
        prob = model.predict_proba(features)[0, 1]
        return prob

    high_risk_prob = get_prediction(high_risk_input)
    medium_risk_prob = get_prediction(medium_risk_input)
    low_risk_prob = get_prediction(low_risk_input)

    print(f"\nPrediction probabilities:")
    print(f"High risk scenario: {high_risk_prob:.4f}")
    print(f"Medium risk scenario: {medium_risk_prob:.4f}")
    print(f"Low risk scenario: {low_risk_prob:.4f}")

    # Verify risk probabilities make sense
    assert high_risk_prob > 0.5, "High risk scenario should have probability > 0.5"
    assert low_risk_prob < 0.5, "Low risk scenario should have probability < 0.5"
    assert high_risk_prob > medium_risk_prob > low_risk_prob, \
        "Risk probabilities don't follow expected pattern"


if __name__ == "__main__":
    main()
