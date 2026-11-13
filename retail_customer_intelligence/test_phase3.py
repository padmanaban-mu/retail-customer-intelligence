import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import clean_transactions, fit_outlier_bounds, apply_outlier_bounds, apply_log1p_transform
from src.feature_engineering import build_customer_features
from src.classification import train_all_models

def test_phase3():
    print("Loading and preparing data...")
    df_raw = load_raw_data()
    df_clean, _ = clean_transactions(df_raw)
    
    cutoff_date = '2011-09-01'
    features_df, target_series = build_customer_features(df_clean, cutoff_date)
    
    cols_to_log = ['Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'Recency', 'AvgInvoiceValue', 'AvgQtyPerInvoice']
    bounds = fit_outlier_bounds(features_df, cols_to_log)
    clipped = apply_outlier_bounds(features_df, bounds)
    features_log = apply_log1p_transform(clipped, cols_to_log)
    
    print("\n--- Training Classifiers ---")
    models, scaler, metrics, feature_names = train_all_models(features_log, target_series)
    
    metrics_df = pd.DataFrame([
        {
            'Model': name, 
            'Accuracy': m['accuracy'], 
            'Precision': m['precision'], 
            'Recall': m['recall'], 
            'F1 (Macro)': m['f1']
        } for name, m in metrics.items() if name != 'data_stats'
    ])
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    test_phase3()
