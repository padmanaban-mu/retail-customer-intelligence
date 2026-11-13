import pandas as pd
import numpy as np

# Load Data
from src.data_loader import load_raw_data
from src.preprocessing import clean_transactions, fit_outlier_bounds, apply_outlier_bounds, apply_log1p_transform
from src.feature_engineering import build_customer_features

raw_df = load_raw_data()
clean_df, _ = clean_transactions(raw_df)
raw_features, target_series = build_customer_features(clean_df, cutoff_date='2011-09-01')

# Train Preprocessing
numeric_cols = ['Recency', 'Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'AvgInvoiceValue', 'AvgQtyPerInvoice']
outlier_bounds = fit_outlier_bounds(raw_features, numeric_cols)
features_clipped = apply_outlier_bounds(raw_features, outlier_bounds)
ml_features = apply_log1p_transform(features_clipped, numeric_cols)
ml_features_unsup = ml_features.drop(columns=['Country_UK'])

# Segment
from src.clustering import get_kmeans_metrics
k_vals, inertias, sils = get_kmeans_metrics(ml_features_unsup, max_k=8)
best_k = k_vals[np.argmax(sils)]
print(f"Chosen K for Segmentation (Best Silhouette): {best_k}")

# Predict
from src.classification import train_all_models, predict_customer
models, scaler, metrics, feature_names = train_all_models(ml_features, target_series)

print("\n--- Cross-Validation Results ---")
for model_name in ["Decision Tree", "Naive Bayes", "SVM (RBF)"]:
    m = metrics[model_name]
    print(f"{model_name}: Acc {m.get('cv_acc_mean',0):.3f}±{m.get('cv_acc_std',0):.3f}, F1 {m.get('cv_f1_mean',0):.3f}±{m.get('cv_f1_std',0):.3f}")

print("\n--- Example Profile Predictions ---")
example_profiles = {
    "Likely Inactive (Churn)": {'Recency': 300, 'Frequency': 1, 'Monetary': 50, 'TotalQuantity': 20, 'UniqueProducts': 5},
    "Average Customer": {'Recency': 30, 'Frequency': 5, 'Monetary': 1000, 'TotalQuantity': 500, 'UniqueProducts': 50},
    "Likely High (VIP)": {'Recency': 5, 'Frequency': 25, 'Monetary': 8000, 'TotalQuantity': 4000, 'UniqueProducts': 150}
}

for name, input_dict in example_profiles.items():
    input_dict['AvgInvoiceValue'] = input_dict['Monetary'] / input_dict['Frequency']
    input_dict['AvgQtyPerInvoice'] = input_dict['TotalQuantity'] / input_dict['Frequency']
    input_dict['Country_UK'] = 1.0
    
    input_df = pd.DataFrame([input_dict])
    input_clipped = apply_outlier_bounds(input_df, outlier_bounds)
    input_log = apply_log1p_transform(input_clipped, numeric_cols)
    final_input_dict = input_log.iloc[0].to_dict()
    
    print(f"\n{name} Profile:")
    for m_name in ["Decision Tree", "Naive Bayes", "SVM (RBF)"]:
        pred, probs = predict_customer(models, scaler, m_name, final_input_dict, feature_names)
        print(f"  {m_name}: {pred} (Probabilities: {probs})")

# Association
from src.preprocessing import clean_for_apriori
from src.association import build_basket_matrix, run_apriori
ap_clean = clean_for_apriori(raw_df)
basket = build_basket_matrix(ap_clean, top_n_products=150, country='United Kingdom')
itemsets, rules = run_apriori(basket, min_support=0.02, min_confidence=0.5, min_lift=1.5)
print(f"\n--- Association Rules ---")
print(f"Total Rules found: {len(rules)}")
