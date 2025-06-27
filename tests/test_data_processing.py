"""
Tests for the data processing pipeline
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import (
    DateFeatureExtractor,
    CustomerAggregator,
    CategoryEncoder,
    process_and_save_data,
    FEATURE_GROUPS
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
        'TransactionStartTime': dates,
        'CustomerId': np.random.randint(1, 11, 100),  # 10 unique customers
        'Amount': np.random.normal(1000, 200, 100),
        'Value': np.abs(np.random.normal(1000, 200, 100)),  # Absolute value for transaction value
        'ProductCategory': np.random.choice(['A', 'B', 'C'], 100),
        'PaymentMethod': np.random.choice(['Credit', 'Debit', 'Cash'], 100),
        'Status': np.random.choice(['Success', 'Failed'], 100),
        'ChannelId': np.random.choice(['Web', 'Android', 'iOS'], 100),
        'CurrencyCode': np.random.choice(['USD', 'EUR', 'GBP'], 100),
        'CountryCode': np.random.choice(['US', 'UK', 'FR'], 100),
        'ProviderId': np.random.choice(['P1', 'P2', 'P3'], 100),
        'ProductId': np.random.choice(['Prod1', 'Prod2', 'Prod3'], 100),
        'PricingStrategy': np.random.choice(['Standard', 'Premium', 'Basic'], 100),
        'SubscriptionId': np.random.randint(1, 6, 100),  # 5 unique subscriptions
        'TransactionId': range(1, 101),
        'BatchId': np.random.randint(1, 21, 100),
        'AccountId': np.random.randint(1, 11, 100),
        'FraudResult': np.random.randint(0, 2, 100)  # Binary target for testing
    }
    
    return pd.DataFrame(data)

def test_date_feature_extractor():
    """Test the DateFeatureExtractor transformer"""
    # Create sample data
    data = pd.DataFrame({
        'TransactionStartTime': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'OtherColumn': [1, 2]
    })
    
    # Apply transformation
    transformer = DateFeatureExtractor()
    result = transformer.fit_transform(data)
    
    # Check if date features were created
    expected_features = FEATURE_GROUPS['temporal_features']
    assert all(col in result.columns for col in expected_features)
    assert 'TransactionStartTime' not in result.columns

def test_customer_aggregator(sample_data):
    """Test the CustomerAggregator transformer"""
    # Apply transformation
    transformer = CustomerAggregator()
    result = transformer.fit_transform(sample_data)
    
    # Check if aggregated features were created
    expected_features = FEATURE_GROUPS['customer_aggregations']
    assert all(col in result.columns for col in expected_features)
    
    # Check if aggregations are correct for a specific customer
    customer_id = sample_data['CustomerId'].iloc[0]
    customer_transactions = sample_data[sample_data['CustomerId'] == customer_id]
    
    assert result[result['CustomerId'] == customer_id]['Transaction_Count'].iloc[0] == len(customer_transactions)
    assert np.isclose(
        result[result['CustomerId'] == customer_id]['Total_Transaction_Amount'].iloc[0],
        customer_transactions['Amount'].sum()
    )
    assert result[result['CustomerId'] == customer_id]['Unique_Product_Categories'].iloc[0] == len(customer_transactions['ProductCategory'].unique())

def test_category_encoder_onehot(sample_data):
    """Test the CategoryEncoder with one-hot encoding"""
    # Apply transformation
    categorical_cols = FEATURE_GROUPS['categorical_features']
    transformer = CategoryEncoder(categorical_columns=categorical_cols, method='onehot')
    result = transformer.fit_transform(sample_data)
    
    # Check if categorical columns were encoded
    for col in categorical_cols:
        # Original column should be dropped
        assert col not in result.columns
        # Check if at least one encoded column exists for each category
        assert any(c.startswith(col + '_') for c in result.columns)

def test_category_encoder_label(sample_data):
    """Test the CategoryEncoder with label encoding"""
    # Apply transformation
    categorical_cols = FEATURE_GROUPS['categorical_features']
    transformer = CategoryEncoder(categorical_columns=categorical_cols, method='label')
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
    for method in ['onehot', 'label']:
        # Process the data with target variable
        target = sample_data['FraudResult']
        result = process_and_save_data(sample_data, temp_output_dir, target=target, categorical_method=method)
        
        # Check if result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check if output files were created
        assert os.path.exists(os.path.join(temp_output_dir, 'features.csv'))
        assert os.path.exists(os.path.join(temp_output_dir, 'raw_for_rfm.csv'))
        assert os.path.exists(os.path.join(temp_output_dir, 'metadata.json'))
        
        # Check raw_for_rfm.csv content
        rfm_data = pd.read_csv(os.path.join(temp_output_dir, 'raw_for_rfm.csv'))
        expected_columns = FEATURE_GROUPS['id_columns'] + FEATURE_GROUPS['rfm_base_features']
        assert all(col in rfm_data.columns for col in expected_columns)
        
        # Check metadata.json content
        with open(os.path.join(temp_output_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            assert 'feature_groups' in metadata
            assert 'transformers' in metadata
            assert 'creation_timestamp' in metadata
            assert 'pipeline_steps' in metadata

def test_feature_groups():
    """Test the feature groups definition"""
    # Check if all required feature groups exist
    required_groups = ['id_columns', 'rfm_base_features', 'categorical_features', 
                      'temporal_features', 'customer_aggregations']
    assert all(group in FEATURE_GROUPS for group in required_groups)
    
    # Check if there are no duplicate features across groups
    all_features = []
    for group in FEATURE_GROUPS.values():
        all_features.extend(group)
    assert len(all_features) == len(set(all_features)), "Duplicate features found across groups"

if __name__ == "__main__":
    pytest.main([__file__])
