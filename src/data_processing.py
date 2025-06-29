"""
Feature Engineering Pipeline for Credit Risk Modeling

This script implements a robust and reproducible data processing pipeline that transforms
raw transaction data into model-ready format using sklearn Pipeline.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Define feature groups for better organization
FEATURE_GROUPS = {
    "id_columns": [
        "TransactionId", "BatchId", "AccountId",
        "SubscriptionId", "CustomerId"
    ],
    "rfm_base_features": [
        "TransactionStartTime", "Amount", "Value"
    ],
    "categorical_features": [
        "ProductCategory", "PaymentMethod", "Status", "ChannelId",
        "CurrencyCode", "CountryCode", "ProviderId", "ProductId",
        "PricingStrategy"
    ],
    "temporal_features": [
        "Transaction_Hour", "Transaction_Day", "Transaction_Month",
        "Transaction_Year", "Transaction_DayOfWeek", "Transaction_WeekOfYear",
        "Is_Weekend", "Is_NightTime"
    ],
    "customer_aggregations": [
        "Total_Transaction_Amount", "Avg_Transaction_Amount",
        "Transaction_Count", "Std_Transaction_Amount",
        "Min_Transaction_Amount", "Max_Transaction_Amount",
        "Total_Value", "Avg_Value", "Unique_Product_Categories",
        "Unique_Channels", "Unique_Providers", "Unique_Currencies",
        "Subscription_Count"
    ]
}


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract date-related features from timestamp columns"""

    def __init__(self, timestamp_column='TransactionStartTime'):
        self.timestamp_column = timestamp_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Convert timestamp to datetime if it's not already
        X[self.timestamp_column] = pd.to_datetime(X[self.timestamp_column])

        # Extract date features
        X['Transaction_Hour'] = X[self.timestamp_column].dt.hour
        X['Transaction_Day'] = X[self.timestamp_column].dt.day
        X['Transaction_Month'] = X[self.timestamp_column].dt.month
        X['Transaction_Year'] = X[self.timestamp_column].dt.year
        X['Transaction_DayOfWeek'] = X[self.timestamp_column].dt.dayofweek
        X['Transaction_WeekOfYear'] = X[self.timestamp_column].dt.isocalendar().week
        X['Is_Weekend'] = X['Transaction_DayOfWeek'].isin([5, 6]).astype(int)
        X['Is_NightTime'] = ((X['Transaction_Hour'] >= 22) | (
            X['Transaction_Hour'] <= 5)).astype(int)

        # Drop original timestamp column
        X = X.drop(columns=[self.timestamp_column])

        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Create aggregate features at customer level"""

    def __init__(self, customer_id='CustomerId', amount_column='Amount', value_column='Value'):
        self.customer_id = customer_id
        self.amount_column = amount_column
        self.value_column = value_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Calculate customer-level aggregations
        aggs = pd.DataFrame()

        # Transaction amount aggregations
        aggs['Total_Transaction_Amount'] = X.groupby(
            self.customer_id)[self.amount_column].sum()
        aggs['Avg_Transaction_Amount'] = X.groupby(
            self.customer_id)[self.amount_column].mean()
        aggs['Transaction_Count'] = X.groupby(self.customer_id)[
            self.amount_column].count()
        aggs['Std_Transaction_Amount'] = X.groupby(
            self.customer_id)[self.amount_column].std()
        aggs['Min_Transaction_Amount'] = X.groupby(
            self.customer_id)[self.amount_column].min()
        aggs['Max_Transaction_Amount'] = X.groupby(
            self.customer_id)[self.amount_column].max()

        # Value-based aggregations
        aggs['Total_Value'] = X.groupby(self.customer_id)[
            self.value_column].sum()
        aggs['Avg_Value'] = X.groupby(self.customer_id)[
            self.value_column].mean()

        # Product category diversity
        aggs['Unique_Product_Categories'] = X.groupby(
            self.customer_id)['ProductCategory'].nunique()

        # Channel diversity
        aggs['Unique_Channels'] = X.groupby(self.customer_id)[
            'ChannelId'].nunique()

        # Provider diversity
        aggs['Unique_Providers'] = X.groupby(self.customer_id)[
            'ProviderId'].nunique()

        # Currency diversity
        aggs['Unique_Currencies'] = X.groupby(self.customer_id)[
            'CurrencyCode'].nunique()

        # Subscription count
        aggs['Subscription_Count'] = X.groupby(self.customer_id)[
            'SubscriptionId'].nunique()

        # Reset index to make customer_id a column
        aggs = aggs.reset_index()

        # Merge back with original data
        X = X.merge(aggs, on=self.customer_id, how='left')

        return X


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables using various methods"""

    def __init__(self, categorical_columns=None, method='onehot', max_categories=10):
        self.categorical_columns = categorical_columns
        self.method = method
        self.max_categories = max_categories
        self.encoders = {}

    def fit(self, X, y=None):
        X = X.copy()

        if self.categorical_columns is None:
            # Define default categorical columns based on Xente variable definitions
            self.categorical_columns = [
                'ProductCategory', 'PaymentMethod', 'Status', 'ChannelId',
                'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                'PricingStrategy'
            ]

        for col in self.categorical_columns:
            if col not in X.columns:
                continue

            if self.method == 'onehot':
                encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown='ignore')
                # Fit on column values reshaped for OneHotEncoder
                encoder.fit(X[[col]])
            elif self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col])

            self.encoders[col] = encoder

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.categorical_columns:
            if col not in X.columns or col not in self.encoders:
                continue

            if self.method == 'onehot':
                # Transform and get feature names
                encoded = self.encoders[col].transform(X[[col]])
                feature_names = self.encoders[col].get_feature_names_out([col])

                # Create DataFrame with encoded values
                encoded_df = pd.DataFrame(
                    encoded, columns=feature_names, index=X.index)

                # Drop original column and concatenate encoded values
                X = X.drop(columns=[col])
                X = pd.concat([X, encoded_df], axis=1)

            elif self.method == 'label':
                X[col] = self.encoders[col].transform(X[col])

        return X


class NumericImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in numeric columns only"""

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer = None

    def fit(self, X, y=None):
        X = X.copy()

        # Get numeric columns only
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_columns) > 0:
            self.imputer = SimpleImputer(strategy=self.strategy)
            self.imputer.fit(X[numeric_columns])

        return self

    def transform(self, X):
        X = X.copy()

        # Get numeric columns only
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_columns) > 0 and self.imputer is not None:
            X[numeric_columns] = self.imputer.transform(X[numeric_columns])

        return X


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Scale numerical features while preserving DataFrame format"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_columns = None

    def fit(self, X, y=None):
        # Store numeric column names
        self.numeric_columns = X.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        if self.numeric_columns:
            # Fit scaler only on numeric columns
            self.scaler.fit(X[self.numeric_columns])
        return self

    def transform(self, X):
        X = X.copy()
        if self.numeric_columns:
            # Transform only numeric columns
            X[self.numeric_columns] = self.scaler.transform(
                X[self.numeric_columns])
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection using Random Forest and Chi-squared importance"""

    def __init__(self, k=20):
        self.k = k
        self.selected_features = None
        self.feature_importances = {}
        self.rf_selector = RandomForestClassifier(
            n_estimators=100, random_state=42)
        self.chi2_selector = SelectKBest(chi2, k=self.k)

    def calculate_feature_importance(self, X, y):
        """Calculate feature importance using multiple methods"""
        importance_scores = {}

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            raise ValueError(
                "No numeric columns available for feature selection")

        X_numeric = X[numeric_cols]

        # Ensure all values are non-negative for chi-squared test
        X_chi2 = X_numeric - X_numeric.min() + 1e-6

        # Random Forest importance
        self.rf_selector.fit(X_numeric, y)
        rf_importance = pd.Series(
            self.rf_selector.feature_importances_, index=numeric_cols)
        importance_scores['random_forest'] = rf_importance

        # Chi-squared importance
        chi2_scores = chi2(X_chi2, y)[0]
        chi2_importance = pd.Series(chi2_scores, index=numeric_cols)
        importance_scores['chi2'] = chi2_importance

        return importance_scores

    def select_features(self, importance_scores):
        """Select top k features based on combined importance"""
        # Normalize each importance score
        normalized_scores = {}
        for method, scores in importance_scores.items():
            if scores.max() > 0:
                normalized_scores[method] = scores / scores.max()
            else:
                normalized_scores[method] = scores

        # Calculate combined importance
        combined_importance = sum(normalized_scores.values())

        # Select top k features
        k = min(self.k, len(combined_importance))
        self.selected_features = combined_importance.nlargest(k).index.tolist()

        return self.selected_features

    def fit(self, X, y=None):
        if y is None:
            raise ValueError(
                "Target variable is required for feature selection")

        # Calculate feature importance using multiple methods
        self.feature_importances = self.calculate_feature_importance(X, y)

        # Select features
        self.selected_features = self.select_features(self.feature_importances)

        return self

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("You must call fit before transform")

        return X[self.selected_features]


def create_feature_pipeline(categorical_method='onehot'):
    """Create the main feature engineering pipeline"""

    pipeline = Pipeline([
        # Extract date features
        ('date_features', DateFeatureExtractor()),

        # Create customer aggregations
        ('customer_aggs', CustomerAggregator()),

        # Handle missing values with mean imputation for numeric features only
        ('imputer', NumericImputer(strategy='mean')),

        # Encode categorical variables
        ('category_encoder', CategoryEncoder(method=categorical_method)),

        # Scale numerical features while preserving DataFrame format
        ('scaler', DataFrameScaler()),

        # Apply feature selection
        ('feature_selector', FeatureSelector(k=20))
    ])

    return pipeline


def save_metadata(pipeline, output_dir):
    """Save pipeline metadata for future reference and API deployment"""
    metadata = {
        "feature_groups": FEATURE_GROUPS,
        "transformers": {
            "date_features": {
                "input_column": "TransactionStartTime",
                "output_features": FEATURE_GROUPS["temporal_features"]
            },
            "customer_aggregations": {
                "input_columns": ["CustomerId", "Amount", "Value"],
                "output_features": FEATURE_GROUPS["customer_aggregations"]
            },
            "categorical_features": {
                "columns": FEATURE_GROUPS["categorical_features"]
            },
            "feature_selection": {
                "n_selected_features": len(pipeline.named_steps['feature_selector'].selected_features),
                "selected_features": pipeline.named_steps['feature_selector'].selected_features,
                "importance_scores": {
                    method: scores.to_dict()
                    for method, scores in pipeline.named_steps['feature_selector'].feature_importances.items()
                }
            }
        },
        "creation_timestamp": datetime.now().isoformat(),
        "pipeline_steps": [step[0] for step in pipeline.steps]
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def process_and_save_data(data, output_dir, target=None, categorical_method='onehot'):
    """Process data and save outputs in structured format"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data needed for RFM calculation
    rfm_data = data[FEATURE_GROUPS["id_columns"] +
                    FEATURE_GROUPS["rfm_base_features"]].copy()
    rfm_data.to_csv(os.path.join(output_dir, 'raw_for_rfm.csv'), index=False)

    # Extract target variable (FraudResult) for feature selection if not provided
    if target is None and 'FraudResult' in data.columns:
        target = data['FraudResult']
        data = data.drop('FraudResult', axis=1)

    # Process features
    pipeline = create_feature_pipeline(categorical_method=categorical_method)
    processed_data = pipeline.fit_transform(data, target)

    # Add back target variable if it was present
    if target is not None:
        processed_data['FraudResult'] = target

    # Save processed features
    processed_data.to_csv(os.path.join(
        output_dir, 'features.csv'), index=False)

    # Save metadata
    save_metadata(pipeline, output_dir)

    return processed_data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        data: Input DataFrame with potential missing values

    Returns:
        DataFrame with handled missing values
    """
    if data.empty:
        raise ValueError("Empty DataFrame provided")

    # Make a copy to avoid modifying the original data
    df = data.copy()

    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding for nominal variables
    and label encoding for ordinal variables.

    Args:
        data: Input DataFrame with categorical features

    Returns:
        DataFrame with encoded categorical features
    """
    if data.empty:
        raise ValueError("Empty DataFrame provided")

    # Make a copy to avoid modifying the original data
    df = data.copy()

    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # One-hot encode nominal variables
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    return df


def preprocess_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features by handling missing values, encoding categorical features,
    and scaling numeric features.

    Args:
        data: Raw input DataFrame

    Returns:
        Preprocessed DataFrame ready for model training
    """
    if data.empty:
        raise ValueError("Empty DataFrame provided")

    # Make a copy to avoid modifying the original data
    df = data.copy()

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical features
    df = encode_categorical_features(df)

    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_cols)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from a CSV file and split into features and target.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Read data
    data = pd.read_csv(file_path)

    # Split features and target
    y = data['is_high_risk']
    X = data.drop('is_high_risk', axis=1)

    # Remove unnamed columns from features
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

    # Ensure y has the correct name
    y.name = 'is_high_risk'

    return X, y


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, model_name: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate model performance using various metrics.

    Args:
        model: Trained model object
        X: Feature DataFrame
        y: Target Series
        model_name: Optional name of the model for logging purposes

    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }

    # Print metrics if model name is provided
    if model_name:
        print(f"\n{model_name} Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    return metrics


def main():
    """Main execution function"""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')

    print("Loading data files...\n")

    # Load variable definitions
    var_defs = pd.read_csv(os.path.join(
        raw_dir, 'Xente_Variable_Definitions.csv'))
    print("Variable definitions:")
    for idx, row in var_defs.iterrows():
        print(f"{row['Column Name']}: {row['Definition']}")

    # Load and process data
    print("\nProcessing data...")
    data = pd.read_csv(os.path.join(raw_dir, 'data.csv'))
    processed_data = process_and_save_data(data, processed_dir)

    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"\nProcessed features: {sorted(processed_data.columns.tolist())}")

    print("\nOutputs saved in:", processed_dir)
    print("Files generated:")
    print("1. features.csv - All engineered features")
    print("2. raw_for_rfm.csv - Raw data for RFM analysis")
    print("3. metadata.json - Feature definitions and pipeline information")


if __name__ == "__main__":
    main()
