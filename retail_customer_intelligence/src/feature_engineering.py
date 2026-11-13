import pandas as pd
import numpy as np

def build_customer_features(df, cutoff_date):
    """
    Builds the customer feature table using strictly pre-cutoff data.
    Computes the target classes using strictly post-cutoff data.
    """
    # Filter out rows without CustomerID for customer-level modules
    df_cust = df.dropna(subset=['CustomerID']).copy()
    
    cutoff = pd.to_datetime(cutoff_date)
    
    # Split data into observation window (features) and target window
    pre_cutoff = df_cust[df_cust['InvoiceDate'] < cutoff]
    post_cutoff = df_cust[df_cust['InvoiceDate'] >= cutoff]
    
    # --- Feature Engineering (Pre-Cutoff) ---
    # Group by CustomerID
    features = pre_cutoff.groupby('CustomerID').agg(
        LastPurchaseDate=('InvoiceDate', 'max'),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum'),
        TotalQuantity=('Quantity', 'sum'),
        UniqueProducts=('StockCode', 'nunique'),
    )
    
    # Calculate Recency in days
    features['Recency'] = (cutoff - features['LastPurchaseDate']).dt.days
    features.drop('LastPurchaseDate', axis=1, inplace=True)
    
    # Calculate Averages
    features['AvgInvoiceValue'] = features['Monetary'] / features['Frequency']
    features['AvgQtyPerInvoice'] = features['TotalQuantity'] / features['Frequency']
    
    # Country UK Flag
    uk_flag = pre_cutoff.groupby('CustomerID')['Country'].apply(lambda x: 1 if (x == 'United Kingdom').any() else 0)
    features['Country_UK'] = uk_flag
    
    # --- Target Variable Creation (Post-Cutoff) ---
    # Calculate future spend per customer
    future_spend = post_cutoff.groupby('CustomerID')['Revenue'].sum().reset_index()
    future_spend.rename(columns={'Revenue': 'FutureSpend'}, inplace=True)
    
    # Merge future spend with the features table
    customer_data = pd.merge(features.reset_index(), future_spend, on='CustomerID', how='left')
    
    # Fill NaN future spend with 0 (Inactive customers)
    customer_data['FutureSpend'] = customer_data['FutureSpend'].fillna(0)
    
    # Define Target Classes
    active_customers = customer_data[customer_data['FutureSpend'] > 0]
    median_active_spend = active_customers['FutureSpend'].median() if len(active_customers) > 0 else 0
    
    def categorize_spend(spend):
        if spend == 0:
            return 'Inactive'
        elif spend <= median_active_spend:
            return 'Low/Medium'
        else:
            return 'High'
            
    customer_data['TargetClass'] = customer_data['FutureSpend'].apply(categorize_spend)
    
    # Clean up structure
    features_df = customer_data.drop(['FutureSpend', 'TargetClass'], axis=1)
    target_series = customer_data[['CustomerID', 'TargetClass']]
    
    return features_df.set_index('CustomerID'), target_series.set_index('CustomerID')
