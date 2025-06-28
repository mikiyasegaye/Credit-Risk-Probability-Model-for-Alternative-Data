"""
RFM Analysis and Target Variable Engineering

This module implements RFM (Recency, Frequency, Monetary) analysis and creates
a proxy target variable for credit risk modeling using customer segmentation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class RFMAnalyzer:
    """Performs RFM analysis and customer segmentation"""
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scalers = {
            'Recency': StandardScaler(),
            'Frequency_Log': StandardScaler(),
            'Monetary_Log': StandardScaler()
        }
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_profiles = None
        self.high_risk_cluster = None
    
    def calculate_rfm(self, data, customer_id='CustomerId', 
                     date_column='TransactionStartTime',
                     amount_column='Amount',
                     snapshot_date=None):
        """
        Calculate RFM metrics for each customer
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Transaction data
        customer_id : str
            Column name for customer ID
        date_column : str
            Column name for transaction date
        amount_column : str
            Column name for transaction amount
        snapshot_date : str or datetime, optional
            Reference date for recency calculation. If None, uses max date in data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with RFM metrics for each customer
        """
        # Convert dates to datetime if needed
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Set snapshot date if not provided
        if snapshot_date is None:
            snapshot_date = data[date_column].max()
        else:
            snapshot_date = pd.to_datetime(snapshot_date)
        
        # Calculate RFM metrics
        rfm = data.groupby(customer_id).agg({
            date_column: lambda x: (snapshot_date - x.max()).days,  # Recency
            customer_id: 'count',  # Frequency
            amount_column: 'sum'   # Monetary
        }).rename(columns={
            date_column: 'Recency',
            customer_id: 'Frequency',
            amount_column: 'Monetary'
        })
        
        return rfm
    
    def preprocess_rfm(self, rfm_data):
        """
        Preprocess RFM data for clustering
        
        Parameters:
        -----------
        rfm_data : pandas.DataFrame
            RFM metrics data
            
        Returns:
        --------
        numpy.ndarray
            Scaled RFM features
        """
        # Create a copy to avoid modifying the original data
        data = rfm_data.copy()
        
        # Handle negative monetary values
        data['Monetary'] = data['Monetary'].abs()
        
        # Log transform frequency and monetary (adding 1 to handle zeros)
        data['Frequency_Log'] = np.log1p(data['Frequency'])
        data['Monetary_Log'] = np.log1p(data['Monetary'])
        
        # Scale each feature independently
        scaled_features = np.zeros((len(data), 3))
        for i, feature in enumerate(['Recency', 'Frequency_Log', 'Monetary_Log']):
            feature_values = data[feature].values.reshape(-1, 1)
            scaled_features[:, i] = self.scalers[feature].fit_transform(feature_values).ravel()
        
        return scaled_features
    
    def identify_high_risk_cluster(self, rfm_data, cluster_labels):
        """
        Identify the high-risk cluster based on RFM profiles
        
        Parameters:
        -----------
        rfm_data : pandas.DataFrame
            Original RFM data
        cluster_labels : numpy.ndarray
            Cluster assignments
            
        Returns:
        --------
        int
            Index of the high-risk cluster
        """
        # Calculate cluster profiles
        self.cluster_profiles = pd.DataFrame()
        for i in range(self.n_clusters):
            mask = cluster_labels == i
            profile = rfm_data[mask].mean()
            self.cluster_profiles[f'Cluster_{i}'] = profile
        
        # High risk cluster has:
        # - High recency (more days since last transaction)
        # - Low frequency
        # - Low monetary value
        
        # Normalize metrics for comparison
        normalized_profiles = self.cluster_profiles.apply(lambda x: (x - x.mean()) / x.std())
        
        # Calculate risk score (higher is riskier)
        risk_scores = (
            normalized_profiles.loc['Recency'] +  # Higher recency is riskier
            -normalized_profiles.loc['Frequency'] +  # Lower frequency is riskier
            -normalized_profiles.loc['Monetary']  # Lower monetary is riskier
        )
        
        # Cluster with highest risk score is the high-risk cluster
        self.high_risk_cluster = risk_scores.argmax()
        
        return self.high_risk_cluster
    
    def fit_predict(self, data, customer_id='CustomerId', 
                   date_column='TransactionStartTime',
                   amount_column='Amount',
                   snapshot_date=None):
        """
        Perform complete RFM analysis and customer segmentation
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Transaction data
        customer_id : str
            Column name for customer ID
        date_column : str
            Column name for transaction date
        amount_column : str
            Column name for transaction amount
        snapshot_date : str or datetime, optional
            Reference date for recency calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with customer IDs and risk labels
        """
        # Calculate RFM metrics
        rfm_data = self.calculate_rfm(
            data, 
            customer_id=customer_id,
            date_column=date_column,
            amount_column=amount_column,
            snapshot_date=snapshot_date
        )
        
        # Preprocess RFM data
        scaled_features = self.preprocess_rfm(rfm_data)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(scaled_features)
        
        # Identify high-risk cluster
        high_risk_cluster = self.identify_high_risk_cluster(rfm_data, cluster_labels)
        
        # Create risk labels
        risk_labels = pd.DataFrame(
            index=rfm_data.index,
            data={
                'is_high_risk': (cluster_labels == high_risk_cluster).astype(int),
                'cluster': cluster_labels
            }
        )
        
        return risk_labels

def create_target_variable(data, output_dir=None):
    """
    Create proxy target variable using RFM analysis and clustering
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data
    output_dir : str, optional
        Directory to save analysis results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with customer IDs and risk labels
    """
    # Initialize RFM analyzer
    rfm_analyzer = RFMAnalyzer(n_clusters=3, random_state=42)
    
    # Perform RFM analysis and clustering
    risk_labels = rfm_analyzer.fit_predict(data)
    
    # If output directory is provided, save analysis results
    if output_dir:
        # Save cluster profiles
        rfm_analyzer.cluster_profiles.to_csv(
            f"{output_dir}/cluster_profiles.csv"
        )
        
        # Save risk labels with index
        risk_labels.to_csv(
            f"{output_dir}/risk_labels.csv"
        )
        
        # Merge risk labels with processed features
        try:
            # Load processed features
            processed_features = pd.read_csv(f"{output_dir}/features.csv")
            
            # Remove any existing unnamed columns
            processed_features = processed_features.loc[:, ~processed_features.columns.str.contains('^Unnamed')]
            
            # Remove any CustomerId columns
            processed_features = processed_features.loc[:, ~processed_features.columns.str.contains('CustomerId')]
            
            # Create a temporary DataFrame with transaction IDs and risk labels
            transaction_risks = pd.DataFrame({
                'TransactionId': processed_features.index,
                'CustomerId': data['CustomerId'].astype(str)
            })
            
            # Map customer risk labels to transactions
            transaction_risks['is_high_risk'] = transaction_risks['CustomerId'].map(
                risk_labels['is_high_risk']
            ).fillna(0).astype(int)
            
            # Set the risk labels in processed features
            processed_features['is_high_risk'] = transaction_risks['is_high_risk']
            
            # Save updated features
            processed_features.to_csv(f"{output_dir}/features.csv")
            
            print("\nMerged risk labels with processed features")
            print("\nRisk label distribution (unique customers):")
            risk_dist = risk_labels['is_high_risk'].value_counts(normalize=True)
            print(risk_dist)
            
            print("\nRisk label distribution (all transactions):")
            trans_dist = processed_features['is_high_risk'].value_counts(normalize=True)
            print(trans_dist)
            
            print("\nFeature statistics by risk group:")
            # Calculate and display statistics for each numeric feature
            numeric_features = processed_features.select_dtypes('number').columns
            numeric_features = [col for col in numeric_features if col != 'is_high_risk']
            
            print("\nSample of features with risk labels:")
            sample_cols = ['is_high_risk'] + numeric_features[:3]
            print(processed_features[sample_cols].head())
            
            # Show statistics for first 5 features by risk group
            for col in numeric_features[:5]:
                print(f"\nStats for {col} by risk group:")
                stats = processed_features.groupby('is_high_risk')[col].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ])
                print(stats)
            
        except FileNotFoundError:
            print("\nWarning: Could not find processed features file to merge risk labels")
        except Exception as e:
            print(f"\nError merging risk labels: {str(e)}")
            print("\nDebug information:")
            print("Risk labels shape:", risk_labels.shape)
            print("Risk labels index sample:", risk_labels.index[:5])
            print("Risk labels distribution:", risk_labels['is_high_risk'].value_counts(normalize=True))
            print("Processed features shape:", processed_features.shape)
            print("Processed features index sample:", processed_features.index[:5])
            raise
    
    return risk_labels

def main():
    """Main execution function"""
    import os
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    print("Loading transaction data...")
    data = pd.read_csv(os.path.join(raw_dir, 'data.csv'))
    
    print("\nPerforming RFM analysis and creating target variable...")
    risk_labels = create_target_variable(data, processed_dir)
    
    print("\nRisk label distribution:")
    print(risk_labels['is_high_risk'].value_counts(normalize=True))
    
    print("\nResults saved in:", processed_dir)

if __name__ == "__main__":
    main() 