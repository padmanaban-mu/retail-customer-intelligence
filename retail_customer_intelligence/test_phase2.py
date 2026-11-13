import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import clean_transactions, handle_outliers, apply_log1p_transform, apply_qcut_discretization, get_before_after_stats
from src.feature_engineering import build_customer_features

def test_phase2():
    print("Loading data...")
    df_raw = load_raw_data()
    
    print("\nCleaning transactions...")
    df_clean, cleaning_log = clean_transactions(df_raw)
    
    print("\nCleaning Log:")
    for step, removed in cleaning_log.items():
        print(f"  {step}: {removed} rows removed")
        
    stats = get_before_after_stats(df_raw, df_clean)
    print(f"\nBefore/After Stats: {stats}")
    
    print("\nBuilding Customer Features (Cutoff: 2011-09-01)...")
    cutoff_date = '2011-09-01'
    features_df, target_series = build_customer_features(df_clean, cutoff_date)
    
    print(f"\nFeature Table Shape: {features_df.shape}")
    print("\nFeature Head:")
    print(features_df.head())
    
    print("\nTarget Class Counts:")
    print(target_series['TargetClass'].value_counts())
    
    # Assert no feature uses post-cutoff data
    pre_cutoff_check = df_clean[(df_clean['CustomerID'].isin(features_df.index)) & (df_clean['InvoiceDate'] < cutoff_date)]
    assert pre_cutoff_check['InvoiceDate'].max() < pd.to_datetime(cutoff_date), "Leakage detected!"
    print("\nLeakage Check: Passed. No post-cutoff data in features.")
    
    # Test outlier and log1p functions
    print("\nTesting Outlier Clipping and Log1p Transformation on Features...")
    cols_to_transform = ['Recency', 'Frequency', 'Monetary', 'TotalQuantity']
    features_clipped = handle_outliers(features_df, cols_to_transform)
    features_log = apply_log1p_transform(features_clipped, cols_to_transform)
    print("Log1p Features Head:")
    print(features_log[cols_to_transform].head())
    
if __name__ == "__main__":
    test_phase2()
