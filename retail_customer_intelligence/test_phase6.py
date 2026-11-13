import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import clean_transactions, fit_outlier_bounds, apply_outlier_bounds, apply_log1p_transform
from src.feature_engineering import build_customer_features
from src.pca_analysis import run_pca
from src.som_analysis import train_som, get_som_visualizations

def test_phase6():
    print("Loading and preparing data...")
    df_raw = load_raw_data()
    df_clean, _ = clean_transactions(df_raw)
    
    cutoff_date = '2011-09-01'
    features_df, _ = build_customer_features(df_clean, cutoff_date)
    
    cols_to_log = ['Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'Recency', 'AvgInvoiceValue', 'AvgQtyPerInvoice']
    bounds = fit_outlier_bounds(features_df, cols_to_log)
    clipped = apply_outlier_bounds(features_df, bounds)
    features_log = apply_log1p_transform(clipped, cols_to_log)
    
    features_unsup = features_log.drop(columns=['Country_UK']) if 'Country_UK' in features_log.columns else features_log
    
    print(f"\nFeature Matrix Shape before PCA/SOM: {features_unsup.shape}")
    
    print("\n--- 1. Testing PCA ---")
    df_pca_all, exp_var_all, cum_var_all, _ = run_pca(features_unsup, n_components=None)
    print("Explained Variance per Component (All):")
    for i, var in enumerate(exp_var_all):
        print(f"  PC{i+1}: {var:.4f}")
        
    print(f"\nCumulative Variance for top 2 components: {cum_var_all[1]:.4f}")
    
    df_pca_2d, _, _, _ = run_pca(features_unsup, n_components=2)
    print("\n2D PCA Projection Head:")
    print(df_pca_2d.head())
    
    print("\n--- 2. Testing SOM (Self-Organizing Map) ---")
    try:
        grid_size = 10
        print(f"Training a {grid_size}x{grid_size} SOM with 5,000 iterations...")
        som, X_scaled = train_som(features_unsup, grid_x=grid_size, grid_y=grid_size, iterations=5000)
        
        u_matrix, hit_map, bmu_coords = get_som_visualizations(som, X_scaled)
        
        print(f"\nU-Matrix Shape: {u_matrix.shape} (Shows cluster boundaries)")
        print(f"Hit Map Shape: {hit_map.shape} (Shows customer density per node)")
        print(f"Number of Best Matching Units Found: {len(bmu_coords)} (Matches customer count)")
        
        print("\nSample Hit Map (Top Left 3x3 Grid):")
        print(hit_map[0:3, 0:3])
        
    except ImportError as e:
        print(f"\nSOM Test Failed: {e}")

if __name__ == "__main__":
    test_phase6()
