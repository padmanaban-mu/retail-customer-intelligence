import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import clean_transactions, fit_outlier_bounds, apply_outlier_bounds, apply_log1p_transform
from src.feature_engineering import build_customer_features
from src.clustering import get_kmeans_metrics, train_kmeans, train_kmedoids, evaluate_clustering, interpret_clusters

def test_phase4():
    print("Loading and preparing data...")
    df_raw = load_raw_data()
    df_clean, _ = clean_transactions(df_raw)
    
    cutoff_date = '2011-09-01'
    features_df, _ = build_customer_features(df_clean, cutoff_date)
    
    # Process features exactly like in app.py
    cols_to_log = ['Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'Recency', 'AvgInvoiceValue', 'AvgQtyPerInvoice']
    bounds = fit_outlier_bounds(features_df, cols_to_log)
    clipped = apply_outlier_bounds(features_df, bounds)
    features_log = apply_log1p_transform(clipped, cols_to_log)
    
    features_unsup = features_log.drop(columns=['Country_UK']) if 'Country_UK' in features_log.columns else features_log
    
    print("\n--- Scanning K from 2 to 8 ---")
    k_vals, inertias, sils = get_kmeans_metrics(features_unsup, max_k=8)
    for k, i, s in zip(k_vals, inertias, sils):
        print(f"K={k}: Inertia={i:.2f}, Silhouette={s:.4f}")
        
    best_k = k_vals[sils.index(max(sils))]
    print(f"\nBest K-Means Silhouette Score is at K={best_k}")
    
    print(f"\n--- Training Models with best K={best_k} ---")
    
    # K-Means
    print("\n1. K-Means Results:")
    kmeans_labels = train_kmeans(features_unsup, k=best_k)
    km_sizes, km_sil, km_profile = evaluate_clustering(features_df, features_unsup, kmeans_labels)
    total_cust = len(features_df)
    km_interp = interpret_clusters(km_profile, km_sizes, total_cust, features_df)
    
    print(f"Silhouette Score: {km_sil:.4f}")
    print(f"Cluster Sizes: {km_sizes}")
    print("Profile Table (Original Units):")
    print(km_profile[['Recency', 'Frequency', 'Monetary']])
    print("Interpretation:")
    for cluster, label in km_interp.items():
        print(f"  Cluster {cluster}: {label}")
        
    # K-Medoids
    print("\n2. K-Medoids (PAM) Results:")
    kmed_labels = train_kmedoids(features_unsup, k=best_k, max_iter=50)
    kmed_sizes, kmed_sil, kmed_profile = evaluate_clustering(features_df, features_unsup, kmed_labels)
    kmed_interp = interpret_clusters(kmed_profile, kmed_sizes, total_cust, features_df)
    
    print(f"Silhouette Score: {kmed_sil:.4f}")
    print(f"Cluster Sizes: {kmed_sizes}")
    print("Profile Table (Original Units):")
    print(kmed_profile[['Recency', 'Frequency', 'Monetary']])
    print("Interpretation:")
    for cluster, label in kmed_interp.items():
        print(f"  Cluster {cluster}: {label}")
        
    print("\n--- K-Means vs K-Medoids Comparison ---")
    print(f"K-Means Silhouette:   {km_sil:.4f}")
    print(f"K-Medoids Silhouette: {kmed_sil:.4f}")

if __name__ == "__main__":
    test_phase4()
