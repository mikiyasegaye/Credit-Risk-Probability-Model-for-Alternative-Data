import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn


def create_sample_data():
    """Create a small sample dataset for testing"""
    # Create sample data with RFM metrics and derived features
    sample_data = {
        'Recency': [5, 30, 100],  # Days since last transaction
        'Frequency': [50, 10, 2],  # Number of transactions
        'Monetary': [10000, 1000, 100],  # Total amount spent
        # Log-transformed frequency
        'Frequency_Log': [np.log1p(50), np.log1p(10), np.log1p(2)],
        # Log-transformed monetary
        'Monetary_Log': [np.log1p(10000), np.log1p(1000), np.log1p(100)],
        'avg_transaction_value': [200, 100, 50],  # Average transaction value
        'transaction_frequency': [10, 5, 1],  # Transactions per month
        'spending_variance': [500, 200, 50],  # Variance in spending
        'max_transaction': [1000, 500, 200],  # Maximum transaction amount
        'min_transaction': [50, 20, 10],  # Minimum transaction amount
        # Total transaction amount
        'total_transactions_amount': [10000, 1000, 100],
        'transaction_consistency': [0.8, 0.5, 0.2],  # Consistency score
        'recent_transaction_trend': [1.2, 0.9, 0.5],  # Recent trend
        'seasonal_pattern': [1, 0, 0],  # Seasonal pattern detected
        # Ratio of weekend transactions
        'weekend_transactions_ratio': [0.3, 0.4, 0.6],
        # Ratio of evening transactions
        'evening_transactions_ratio': [0.2, 0.3, 0.5],
        'transaction_day_variance': [2, 5, 10],  # Variance in transaction days
        # Growth in transaction amounts
        'transaction_amount_growth': [0.1, -0.1, -0.3],
        # Average days between transactions
        'days_between_transactions': [7, 15, 30],
        'inactive_periods': [1, 3, 5],  # Number of inactive periods
        'cluster': [0, 1, 2]  # Cluster assignment
    }
    return pd.DataFrame(sample_data)


def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")

    # Create sample data
    print("\nCreating sample data...")
    sample_data = create_sample_data()
    print("\nSample data (first few columns):")
    print(sample_data[['Recency', 'Frequency', 'Monetary',
          'avg_transaction_value', 'transaction_frequency']].to_string())

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
                    'metrics.roc_auc', ascending=False).iloc[0]
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
                        f"  Recency: {sample_data.iloc[i]['Recency']} days since last transaction")
                    print(
                        f"  Frequency: {sample_data.iloc[i]['Frequency']} transactions")
                    print(
                        f"  Monetary: ${sample_data.iloc[i]['Monetary']:.2f} total spent")
                    print(
                        f"  Avg Transaction: ${sample_data.iloc[i]['avg_transaction_value']:.2f}")
                    print(
                        f"Prediction: {'High Risk' if predictions[i] == 1 else 'Low Risk'}")
                    print(f"Probability of High Risk: {probabilities[i]:.2%}")

                break  # We found and used the best model, so we can stop
            else:
                print("No runs found in this experiment")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
