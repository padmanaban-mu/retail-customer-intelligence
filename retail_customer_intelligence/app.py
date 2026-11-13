import streamlit as st
import pandas as pd
import numpy as np

# Set page config FIRST
st.set_page_config(page_title="Retail Customer Intelligence", layout="wide", page_icon="🛒")

# Imports
from src.data_loader import load_raw_data
from src.preprocessing import clean_transactions, handle_outliers, apply_log1p_transform, get_before_after_stats, fit_outlier_bounds, apply_outlier_bounds
from src.feature_engineering import build_customer_features
from src.classification import train_all_models, predict_customer
from src.clustering import get_kmeans_metrics, train_kmeans, train_kmedoids, evaluate_clustering, interpret_clusters
from src.association import build_basket_matrix, run_apriori
from src.pca_analysis import run_pca
from src.som_analysis import train_som, get_som_visualizations, MINISOM_AVAILABLE
import src.visualization as viz
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Caching Data and Models ---
@st.cache_data
def load_and_clean_data():
    raw_df = load_raw_data()
    clean_df, log = clean_transactions(raw_df)
    stats = get_before_after_stats(raw_df, clean_df)
    return raw_df, clean_df, log, stats

@st.cache_data
def generate_features(clean_df, cutoff_date):
    features_df, target_series = build_customer_features(clean_df, cutoff_date)
    return features_df, target_series

@st.cache_data
def preprocess_features_for_ml(features_df):
    cols = ['Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'Recency', 'AvgInvoiceValue', 'AvgQtyPerInvoice']
    bounds = fit_outlier_bounds(features_df, cols)
    clipped = apply_outlier_bounds(features_df, bounds)
    transformed = apply_log1p_transform(clipped, cols)
    return transformed, bounds

@st.cache_resource
def get_trained_models(features_df, target_series):
    return train_all_models(features_df, target_series)

# --- Navigation ---
PAGES = [
    "Dashboard", 
    "Dataset & Preprocessing", 
    "Customer Behavior Prediction", 
    "Model Comparison", 
    "Customer Segmentation", 
    "Product Association Analysis", 
    "PCA Visualization", 
    "SOM Visualization", 
    "About Project"
]

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", PAGES)

# --- Load Data Pipeline safely ---
try:
    raw_df, clean_df, clean_log, clean_stats = load_and_clean_data()
    CUTOFF_DATE = '2011-09-01'
    raw_features, target_series = generate_features(clean_df, CUTOFF_DATE)
    ml_features, outlier_bounds = preprocess_features_for_ml(raw_features)
    ml_features_unsup = ml_features.drop(columns=['Country_UK']) if 'Country_UK' in ml_features.columns else ml_features
    models, scaler, metrics, feature_names = get_trained_models(ml_features, target_series)
except Exception as e:
    st.error(f"Critical Error Loading Data Pipeline: {e}")
    st.stop()

# --- Page: Dashboard ---
if selection == "Dashboard":
    st.title("🛒 Retail Customer Intelligence Dashboard")
    st.info("**WHAT:** A high-level overview of the dataset.\n**WHY:** To understand the scale of the business.\n**RESULT:** Key Performance Indicators (KPIs) and temporal trends.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{clean_stats['rows_after']:,}")
    col2.metric("Total Customers", f"{clean_stats['customers_after']:,}")
    col3.metric("Total Products", f"{clean_df['StockCode'].nunique():,}")
    col4.metric("Total Revenue", f"£{clean_stats['revenue_after']:,.2f}")
    
    st.caption("*Note: All dashboard totals reflect the dataset after the data cleaning phase.*")
    
    st.write(f"**Date Range:** {clean_df['InvoiceDate'].min().date()} to {clean_df['InvoiceDate'].max().date()}")
    
    st.subheader("Revenue Over Time")
    st.pyplot(viz.plot_revenue_over_time(clean_df))
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 5 Countries by Revenue")
        country_rev = clean_df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5)
        st.dataframe(country_rev, use_container_width=True)
    with col_b:
        st.subheader("Top 5 Products by Quantity Sold")
        prod_qty = clean_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(5)
        st.dataframe(prod_qty, use_container_width=True)

# --- Page: Dataset & Preprocessing ---
elif selection == "Dataset & Preprocessing":
    st.title("🧹 Dataset & Preprocessing")
    st.info("**WHAT:** Data cleaning, outlier clipping, and log transformation.\n**WHY:** Skewed features and outliers distort distance/probability-based models, so we clip and log-transform; models don't need 'normally-distributed' data.\n**RESULT:** A robust, leakage-free feature table ready for prediction.")
    
    st.write("**Missing Values Logic:** Rows with missing CustomerID are dropped for customer-level models (like Segmentation and Prediction), but kept for Market Basket Analysis (Apriori) because we still care about what items were bought together in those invoices.")
    
    st.subheader("Raw Data Profiling")
    prof_df = pd.DataFrame({
        'Metric': ['Total Rows', 'Missing CustomerID', 'Missing Description', 'Duplicate Rows', 'Cancelled Invoices', 'Negative/Zero Quantity', 'Invalid Prices'],
        'Value': [len(raw_df), raw_df['CustomerID'].isnull().sum(), raw_df['Description'].isnull().sum(), 
                  raw_df.duplicated().sum(), raw_df['InvoiceNo'].astype(str).str.startswith('C').sum(),
                  (raw_df['Quantity'] <= 0).sum(), (raw_df['UnitPrice'] <= 0).sum()]
    })
    st.dataframe(prof_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cleaning Log (Rows Removed)")
        for k, v in clean_log.items():
            st.write(f"- **{k}:** {v:,}")
            
    with col2:
        st.subheader("Before / After Summary")
        st.write(f"- Rows: {clean_stats['rows_before']:,} -> {clean_stats['rows_after']:,}")
        st.write(f"- Revenue: £{clean_stats['revenue_before']:,.0f} -> £{clean_stats['revenue_after']:,.0f}")
        
    st.subheader("Target Definition")
    st.info(f"**Cutoff Date:** {CUTOFF_DATE}\n- Features (Recency, Frequency, Monetary, etc.) are computed **strictly before** this date.\n- The Target class (Inactive, Low/Medium, High) is based on spend **strictly after** this date.\nThis prevents data leakage.")
    st.write(target_series['TargetClass'].value_counts().reset_index().rename(columns={'index':'Class', 'TargetClass':'Count'}))
    
    st.subheader("Handling Outliers & Skewness")
    st.write("Used **percentile clipping** (1st to 99th percentile) to remove extreme outliers, followed by **log1p transformation** to handle right-tailed skewness.")
    
    # Compute Outlier stats
    outlier_stats = []
    numeric_cols = ['Recency', 'Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'AvgInvoiceValue', 'AvgQtyPerInvoice']
    for col in numeric_cols:
        series = raw_features[col]
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower_iqr, upper_iqr = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        out_iqr = ((series < lower_iqr) | (series > upper_iqr)).sum()
        
        p01, p99 = series.quantile(0.01), series.quantile(0.99)
        clipped = ((series < p01) | (series > p99)).sum()
        
        outlier_stats.append({
            'Feature': col,
            'IQR Lower': lower_iqr,
            'IQR Upper': upper_iqr,
            'Outside IQR': out_iqr,
            'Below p1': (series < p01).sum(),
            'Above p99': (series > p99).sum(),
            'Total Clipped': clipped
        })
    st.dataframe(pd.DataFrame(outlier_stats).style.format({'IQR Lower': '{:.2f}', 'IQR Upper': '{:.2f}'}), use_container_width=True)
    
    st.pyplot(viz.plot_before_after_histogram(raw_features, ml_features, column='Monetary'))
    st.pyplot(viz.plot_before_after_histogram(raw_features, ml_features, column='Frequency'))
    
    st.subheader("Discretization (qcut bins)")
    st.write("For interpretability, we bin continuous variables into Low, Medium, and High categories.")
    
    def display_qcut(col):
        binned, edges = pd.qcut(raw_features[col], q=3, retbins=True, duplicates='drop')
        n_bins = len(edges) - 1
        
        if n_bins == 3:
            labels = ['Low', 'Medium', 'High']
        elif n_bins == 2:
            labels = ['Low', 'High']
        else:
            labels = ['All']
            
        binned = pd.cut(raw_features[col], bins=edges, labels=labels, include_lowest=True)
        counts = binned.value_counts().to_dict()
        
        st.write(f"**{col} Bins:** {np.round(edges, 2)}")
        counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
        st.write(f"- {counts_str}")
    display_qcut('Recency')
    display_qcut('Frequency')
    display_qcut('Monetary')
    
    st.subheader(f"Customer feature table (before clipping and log transform)")
    st.dataframe(raw_features.head(10), use_container_width=True)
    
    st.subheader(f"After clipping, log1p and scaling")
    st.dataframe(ml_features.head(10), use_container_width=True)

# --- Page: Customer Behavior Prediction ---
elif selection == "Customer Behavior Prediction":
    st.title("🔮 Customer Behavior Prediction")
    st.info("**WHAT:** Predicts a customer's future spend category.\n**WHY:** To identify which customers will churn (Inactive) vs become VIPs (High).\n**RESULT:** A predicted class and confidence probabilities.")
    
    st.write("### Test a Prediction")
    
    # Extract test set customers from metrics
    test_counts = metrics.get('data_stats', {}).get('test_counts', {})
    
    # Create example profiles
    example_profiles = {
        "Manual Input": None,
        "Likely Inactive (Churn)": {'Recency': 300, 'Frequency': 1, 'Monetary': 50, 'TotalQuantity': 20, 'UniqueProducts': 5},
        "Average Customer": {'Recency': 30, 'Frequency': 5, 'Monetary': 1000, 'TotalQuantity': 500, 'UniqueProducts': 50},
        "Likely High (VIP)": {'Recency': 5, 'Frequency': 25, 'Monetary': 8000, 'TotalQuantity': 4000, 'UniqueProducts': 150}
    }
    
    test_mode = st.radio("Input Method:", ["Select Existing Customer", "Use Example Profile", "Enter Manual Values"])
    
    input_dict = {}
    is_valid = True
    
    if test_mode == "Select Existing Customer":
        # Only select customers that are in the test set or specify if they are not
        # To keep it simple, we just pick from all and label them
        cust_id_int = st.selectbox("Select Customer ID", raw_features.index.astype(int))
        cust_id = float(cust_id_int)
        actual_class = target_series.loc[cust_id]['TargetClass']
        # Note: We can't strictly know if it was in the test set easily without saved indices, so we'll just say "Historical Customer"
        st.write(f"**Actual Future Target Class:** {actual_class}")
        input_dict = raw_features.loc[cust_id].to_dict()
        st.json(input_dict)
    elif test_mode == "Use Example Profile":
        profile_choice = st.selectbox("Select Example:", list(example_profiles.keys())[1:])
        input_dict = example_profiles[profile_choice]
        input_dict['AvgInvoiceValue'] = input_dict['Monetary'] / input_dict['Frequency']
        input_dict['AvgQtyPerInvoice'] = input_dict['TotalQuantity'] / input_dict['Frequency']
        input_dict['Country_UK'] = 1.0
        st.json(input_dict)
    else:
        col1, col2 = st.columns(2)
        with col1:
            input_dict['Frequency'] = st.number_input("Frequency (Invoices)", min_value=1, value=5)
            input_dict['Monetary'] = st.number_input("Monetary (Total Spend £)", min_value=0.0, value=1000.0)
            input_dict['TotalQuantity'] = st.number_input("Total Quantity Bought", min_value=1, value=500)
            input_dict['UniqueProducts'] = st.number_input("Unique Products Bought", min_value=1, value=50)
        with col2:
            input_dict['Recency'] = st.number_input("Recency", min_value=0, value=30, help="Days since last purchase, measured as of the cutoff date (2011-09-01).")
            st.caption("Days since last purchase, measured as of the cutoff date (2011-09-01).")
            
            input_dict['AvgInvoiceValue'] = input_dict['Monetary'] / max(input_dict['Frequency'], 1)
            input_dict['AvgQtyPerInvoice'] = input_dict['TotalQuantity'] / max(input_dict['Frequency'], 1)
            
            st.text_input("Avg Invoice Value (£) (Computed)", f"{input_dict['AvgInvoiceValue']:.2f}", disabled=True)
            st.text_input("Avg Qty per Invoice (Computed)", f"{input_dict['AvgQtyPerInvoice']:.2f}", disabled=True)
            
            input_dict['Country_UK'] = st.selectbox("From UK?", [1.0, 0.0])
            
    if is_valid:
        # Apply identical transformation to the input before feeding to model
        try:
            input_df = pd.DataFrame([input_dict])
            input_clipped = apply_outlier_bounds(input_df, outlier_bounds)
            cols_to_log = ['Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'Recency', 'AvgInvoiceValue', 'AvgQtyPerInvoice']
            input_log = apply_log1p_transform(input_clipped, cols_to_log)
            final_input_dict = input_log.iloc[0].to_dict()
            
            st.write("---")
            baseline_acc = metrics['Baseline (Majority Class)']['accuracy']
            
            def render_prediction(model_name):
                pred, probs = predict_customer(models, scaler, model_name, final_input_dict, feature_names)
                m = metrics[model_name]
                st.write(f"**Test Accuracy:** {m['accuracy']:.1%} | **Macro F1:** {m['f1']:.1%} *(Baseline Acc: {baseline_acc:.1%})*")
                
                if test_mode == "Select Existing Customer":
                    actual = target_series.loc[cust_id]['TargetClass']
                    color = "green" if pred == actual else "red"
                    st.markdown(f"**Predicted:** <span style='color:{color}'>{pred}</span> vs **Actual:** {actual}", unsafe_allow_html=True)
                else:
                    st.success(f"**Predicted Class:** {pred}")
                    
                if probs:
                    st.write("Probabilities:")
                    for k, v in probs.items():
                        st.write(f"- {k}: {v:.1%}")
                if model_name == "SVM (RBF)":
                    st.caption("SVM probabilities come from an extra calibration step and are less reliable than the predicted class.")
                return pred, probs
                
            if st.button("Compare all three models"):
                results = []
                for m_name in ["Decision Tree", "Naive Bayes", "SVM (RBF)"]:
                    p, probs = predict_customer(models, scaler, m_name, final_input_dict, feature_names)
                    conf = max(probs.values()) if probs else None
                    results.append({
                        "Model": m_name,
                        "Predicted Class": p,
                        "Confidence": f"{conf:.1%}" if conf else "N/A"
                    })
                st.table(pd.DataFrame(results))
                
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.subheader("Decision Tree")
                render_prediction("Decision Tree")
            with col_m2:
                st.subheader("Naive Bayes")
                render_prediction("Naive Bayes")
            with col_m3:
                st.subheader("SVM (RBF)")
                render_prediction("SVM (RBF)")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- Page: Model Comparison ---
elif selection == "Model Comparison":
    st.title("📊 Model Comparison")
    st.info("**WHAT:** Evaluates model performance against a baseline.\n**WHY:** To prove the models actually learned real patterns rather than guessing.\n**RESULT:** Accuracy, Precision, Recall, and Confusion Matrices.")
    
    st.write("### Model Explanations")
    st.write("- **Decision Tree:** Learns decision rules (if-then logic) by splitting the data. Easy to interpret but prone to overfitting.")
    st.write("- **Naive Bayes:** Calculates probabilities with a feature-independence assumption. Fast and works well with independent features.")
    st.write("- **SVM (RBF):** Finds a maximum-margin boundary using scaled features in a higher-dimensional space. Good for complex boundaries.")
    
    st.write("### Class Distribution (Train vs Test)")
    st.write("Baseline model always predicts the majority class ('Inactive').")
    train_counts = metrics.get('data_stats', {}).get('train_counts', {})
    test_counts = metrics.get('data_stats', {}).get('test_counts', {})
    dist_df = pd.DataFrame({'Train Set': train_counts, 'Test Set': test_counts}).fillna(0).astype(int)
    st.dataframe(dist_df.T, use_container_width=True)
    
    st.subheader("Performance Metrics")
    sel_metric = st.selectbox("Highlight Metric:", ["Accuracy", "Precision", "Recall", "F1 (Macro)"])
    
    metrics_list = []
    for name, m in metrics.items():
        if name == 'data_stats': continue
        metrics_list.append({
            'Model': name, 
            'Accuracy': m['accuracy'], 
            'Precision': m['precision'], 
            'Recall': m['recall'], 
            'F1 (Macro)': m['f1'],
            'CV Acc (Mean±Std)': f"{m.get('cv_acc_mean', 0):.1%} ± {m.get('cv_acc_std', 0):.1%}" if 'cv_acc_mean' in m else "N/A",
            'CV F1 (Mean±Std)': f"{m.get('cv_f1_mean', 0):.1%} ± {m.get('cv_f1_std', 0):.1%}" if 'cv_f1_mean' in m else "N/A",
            'Train Acc': m.get('train_accuracy', 0)
        })
    metrics_df = pd.DataFrame(metrics_list)
    
    # Custom bar chart for selected metric
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=metrics_df, x='Model', y=sel_metric, hue='Model', palette='viridis', legend=False, ax=ax)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(metrics_df[sel_metric]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Highlight the selected metric column
    st.dataframe(metrics_df.style.highlight_max(subset=[sel_metric], color='lightgreen', axis=0), use_container_width=True)
    
    # Overfitting Check
    st.subheader("Train vs Test Accuracy (Overfitting Check)")
    dt_m = metrics.get('Decision Tree', {})
    svm_m = metrics.get('SVM (RBF)', {})
    dt_diff = dt_m.get('train_accuracy', 0) - dt_m.get('accuracy', 0)
    svm_diff = svm_m.get('train_accuracy', 0) - svm_m.get('accuracy', 0)
    st.write(f"- **Decision Tree:** Train {dt_m.get('train_accuracy', 0):.1%} vs Test {dt_m.get('accuracy', 0):.1%} (Diff: {dt_diff:.1%})")
    st.write(f"- **SVM (RBF):** Train {svm_m.get('train_accuracy', 0):.1%} vs Test {svm_m.get('accuracy', 0):.1%} (Diff: {svm_diff:.1%})")
    if dt_diff > 0.05 or svm_diff > 0.05:
        st.write("*Note: A large difference (>5%) indicates the model is memorizing the training data (overfitting) rather than generalizing.*")
    else:
        st.write("*Note: Train and test accuracies are close, indicating good generalization without severe overfitting.*")
    
    st.subheader("Confusion Matrices")
    class_names = models['Decision Tree'].classes_ if 'Decision Tree' in models else None
    
    # Create dict without data_stats
    plot_metrics = {k: v for k, v in metrics.items() if k != 'data_stats'}
    st.pyplot(viz.plot_confusion_matrices(plot_metrics, class_names=class_names))
    
    # Note below confusion matrices
    st.write("*Note: Low/Medium is the hardest class and is often predicted as Inactive; differences between models are within noise on this test set.*")
    st.subheader("Classification Reports")
    for name in ["Decision Tree", "Naive Bayes", "SVM (RBF)"]:
        if name in metrics:
            with st.expander(f"{name} Report"):
                report_df = pd.DataFrame(metrics[name]['classification_report']).T
                st.dataframe(report_df.style.format('{:.3f}'))
                
    if 'Decision Tree' in metrics and 'feature_importances' in metrics['Decision Tree']:
        st.subheader("Decision Tree Feature Importance")
        fi = metrics['Decision Tree']['feature_importances']
        fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=fi_df, x='Importance', y='Feature', color='coral', ax=ax2)
        ax2.set_title("Which features did the Decision Tree rely on the most?")
        plt.tight_layout()
        st.pyplot(fig2)
        
    if 'tree_plot' in metrics.get('Decision Tree', {}):
        st.subheader("Decision Tree Logic (Depth Limited to 3)")
        st.pyplot(metrics['Decision Tree']['tree_plot'])

# --- Page: Customer Segmentation ---
elif selection == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.info("**WHAT:** Unsupervised grouping of customers.\n**WHY:** To identify distinct marketing personas without using pre-defined labels.\n**RESULT:** K-Means and K-Medoids clusters with interpretations.")
    
    alg = st.radio("Select Algorithm:", ["K-Means", "K-Medoids (PAM)"])
    
    k_vals, inertias, sils = get_kmeans_metrics(ml_features_unsup, max_k=8)
    st.pyplot(viz.plot_elbow_silhouette(k_vals, inertias, sils))
    best_k = k_vals[np.argmax(sils)]
    
    st.write(f"**Note:** Silhouette is highest at K={best_k}, but K=4 was chosen for interpretability; scores for K=4 and K=5 are nearly equal.")
        
    k_choice = st.slider("Select K (Number of Clusters)", min_value=2, max_value=8, value=4)
    st.session_state['seg_alg'] = alg
    st.session_state['seg_k'] = k_choice
    
    with st.spinner("Training clustering model..."):
        if alg == "K-Means":
            labels = train_kmeans(ml_features_unsup, k=k_choice)
        else:
            labels = train_kmedoids(ml_features_unsup, k=k_choice, max_iter=30)
            
        sizes, sil, profile = evaluate_clustering(raw_features, ml_features_unsup, labels)
        total_cust = len(raw_features)
        interp = interpret_clusters(profile, sizes, total_cust, raw_features)
        
    st.write("Labels are our interpretation of the cluster statistics; the algorithm itself does not know what a high-value customer is.")
    
    # K-Means vs K-Medoids comparison section
    st.subheader("K-Means vs K-Medoids Comparison")
    st.write("K-Means represents a cluster by its mean (a point that may not exist), while K-Medoids uses an actual customer as the centre, which makes it less sensitive to outliers.")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.write("**K-Means**")
        km_l = train_kmeans(ml_features_unsup, k=k_choice)
        km_s, km_sil, _ = evaluate_clustering(raw_features, ml_features_unsup, km_l)
        st.write(f"- Silhouette: {km_sil:.4f}")
        st.write(f"- Smallest Cluster: {min(km_s.values())}")
        st.write(f"- Largest Cluster: {max(km_s.values())}")
    with col_c2:
        st.write("**K-Medoids**")
        kmed_l = train_kmedoids(ml_features_unsup, k=k_choice, max_iter=30)
        kmed_s, kmed_sil, _ = evaluate_clustering(raw_features, ml_features_unsup, kmed_l)
        st.write(f"- Silhouette: {kmed_sil:.4f}")
        st.write(f"- Smallest Cluster: {min(kmed_s.values())}")
        st.write(f"- Largest Cluster: {max(kmed_s.values())}")
        
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(km_l, kmed_l)
    st.write(f"**Adjusted Rand Index between the two labelings:** {ari:.4f}")
    
    st.subheader("Cluster Profiles (Average values per cluster)")
    # Format the profile table nicely
    disp_profile = profile[['Recency', 'Frequency', 'Monetary']].copy()
    disp_profile['Size'] = disp_profile.index.map(sizes)
    disp_profile['Interpretation'] = disp_profile.index.map(interp)
    st.dataframe(disp_profile.style.format({'Recency': '{:.1f}', 'Frequency': '{:.1f}', 'Monetary': '£{:.2f}'}), use_container_width=True)
    
    st.subheader("2D Visualization (via PCA)")
    df_pca, _, _, _ = run_pca(ml_features_unsup, n_components=2)
    st.pyplot(viz.plot_pca_scatter(df_pca, labels=labels))
    st.write("PCA is used only to draw the clusters in 2D; clustering was done on all features.")

# --- Page: Product Association ---
elif selection == "Product Association Analysis":
    st.title("🛍️ Product Association Analysis")
    st.info("**WHAT:** Apriori algorithm for Market Basket Analysis.\n**WHY:** To discover which products are frequently bought together.\n**RESULT:** Association rules indicating cross-selling opportunities.")
    
    st.write("Using Invoice-level data for the United Kingdom.")
    
    col1, col2, col3 = st.columns(3)
    min_sup = col1.slider("Minimum Support", 0.01, 0.10, 0.02, 0.01)
    st.caption("**Support:** Fraction of transactions containing the itemset.")
    min_conf = col2.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)
    st.caption("**Confidence:** Likelihood of buying consequent given antecedent.")
    min_lift = col3.slider("Minimum Lift", 1.0, 10.0, 1.5, 0.5)
    st.caption("**Lift:** How much more likely the items are bought together than if independent.")
    
    with st.spinner("Mining rules (this can take a moment)..."):
        # Cache the basket building
        @st.cache_data
        def get_basket():
            from src.preprocessing import clean_for_apriori
            ap_clean = clean_for_apriori(raw_df)
            return build_basket_matrix(ap_clean, top_n_products=150, country='United Kingdom')
        
        basket = get_basket()
        itemsets, rules = run_apriori(basket, min_support=min_sup, min_confidence=min_conf, min_lift=min_lift)
        
    st.write(f"**Data Used:** {basket.shape[0]:,} invoices and {basket.shape[1]:,} frequent products.")
    
    if rules.empty:
        st.warning("No rules found for these thresholds. Try lowering the requirements (e.g. support = 0.01 or confidence = 0.2).")
    else:
        st.success(f"Found {len(itemsets)} frequent itemsets and {len(rules)} rules!")
        
        st.subheader("Frequent Itemsets")
        itemsets['Itemset size'] = itemsets['itemsets'].apply(lambda x: len(x))
        itemsets['itemsets'] = itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
        show_single = st.checkbox("Include single items", value=False)
        if not show_single:
            display_itemsets = itemsets[itemsets['Itemset size'] >= 2]
        else:
            display_itemsets = itemsets
        st.dataframe(display_itemsets.sort_values('support', ascending=False).head(10), use_container_width=True)
        
        st.subheader("Association Rules")
        # Format for display
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False)
        st.dataframe(display_rules.style.format({'support': '{:.4f}', 'confidence': '{:.4f}', 'lift': '{:.3f}'}), use_container_width=True)
        
        st.subheader("Top 10 Rules by Lift")
        top_rules = display_rules.head(10).copy()
        top_rules['Rule'] = top_rules['antecedents'] + " -> " + top_rules['consequents']
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_rules, x='lift', y='Rule', color='skyblue', ax=ax)
        ax.set_title("Top Rules by Lift")
        plt.tight_layout()
        st.pyplot(fig)
        
        top_rule = display_rules.iloc[0]
        st.write(f"**Example:** If a customer buys **{top_rule['antecedents']}**, they are **{top_rule['lift']:.1f}x** more likely to buy **{top_rule['consequents']}** than a random customer.")

# --- Page: PCA Visualization ---
elif selection == "PCA Visualization":
    st.title("📉 PCA Visualization")
    st.info("**WHAT:** Principal Component Analysis.\n**WHY:** To reduce the high-dimensional feature space into a 2D space for visualization.\n**RESULT:** Explained variance metrics and a 2D scatter plot.")
    
    st.write("PCA transforms many correlated features into fewer components while preserving as much variance as possible. It is a dimensionality reduction technique, not a prediction algorithm.")
    st.write(f"**Original features: {ml_features_unsup.shape[1]} -> components kept: 2**")
    
    df_pca_all, exp_var, cum_var, pca_all = run_pca(ml_features_unsup, n_components=None)
    
    comp_to_keep = st.slider("Components to keep", min_value=1, max_value=len(exp_var), value=2)
    st.write(f"Cumulative Variance for {comp_to_keep} components: {cum_var[comp_to_keep-1]:.2%}")
    
    n_90 = next(i for i, v in enumerate(cum_var) if v >= 0.90) + 1
    n_95 = next(i for i, v in enumerate(cum_var) if v >= 0.95) + 1
    st.write(f"Components needed for 90% variance: **{n_90}** | For 95% variance: **{n_95}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Features per Component (Loadings)")
        loadings = pd.DataFrame(pca_all.components_.T, columns=[f'PC{i+1}' for i in range(len(exp_var))], index=ml_features_unsup.columns)
        pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False).head(3)
        st.dataframe(loadings[['PC1', 'PC2']].style.background_gradient(cmap='coolwarm'), use_container_width=True)
        top_pc1_feats = ", ".join(pc1_loadings.index.tolist())
        st.write(f"**PC1 Interpretation:** Strongly driven by {top_pc1_feats}.")
    with col2:
        st.subheader("Cumulative Variance Chart")
        var_df = pd.DataFrame({'Component': [f'PC{i+1}' for i in range(len(exp_var))], 'Cumulative Variance': cum_var})
        st.line_chart(var_df.set_index('Component')['Cumulative Variance'])
        
    st.subheader("2D Projection")
    # Get labels from Segmentation page if available, else K-Means best K
    k_choice = st.session_state.get('seg_k', None)
    alg_choice = st.session_state.get('seg_alg', 'K-Means')
    if k_choice is None:
        k_vals, _, sils = get_kmeans_metrics(ml_features_unsup, max_k=8)
        k_choice = k_vals[np.argmax(sils)]
        
    if alg_choice == "K-Means":
        cluster_labels = train_kmeans(ml_features_unsup, k=k_choice)
    else:
        cluster_labels = train_kmedoids(ml_features_unsup, k=k_choice, max_iter=30)
        
    color_mode = st.radio("Color points by:", [f"Cluster ({alg_choice}, K={k_choice})", "Actual Future Behaviour Class (Exploration Only)"])
    
    df_pca_2d, _, _, _ = run_pca(ml_features_unsup, n_components=2)
    
    if color_mode.startswith("Cluster"):
        # We need interpretations to create a readable legend
        sizes, _, profile = evaluate_clustering(raw_features, ml_features_unsup, cluster_labels)
        interp = interpret_clusters(profile, sizes, len(raw_features), raw_features)
        
        df_pca_2d['Label'] = [interp[lbl] for lbl in cluster_labels]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_pca_2d, x='PC1', y='PC2', hue='Label', palette='tab10', alpha=0.7, ax=ax)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        df_pca_2d['Label'] = target_series['TargetClass'].values
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_pca_2d, x='PC1', y='PC2', hue='Label', palette='Set1', alpha=0.7, ax=ax)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

# --- Page: SOM Visualization ---
# --- Page: SOM Visualization ---
elif selection == "SOM Visualization":
    st.title("🧠 SOM Visualization")
    st.info("**WHAT:** Self-Organizing Map (Neural Network).\n**WHY:** To map high-dimensional customer data into a 2D grid while preserving topological properties (similar customers are placed near each other).\n**RESULT:** Distance matrices and density hit maps.")
    
    st.warning("**Note:** SOM is an **unsupervised** learning algorithm. It does not predict future behaviour; it only maps existing behaviour into neighborhoods based on similarity.")
    st.write("**How to read these maps:**\n- **U-Matrix:** dark U-matrix cells are boundaries between groups, light regions are groups of similar customers.\n- **Hit Map:** Shows the exact number of customers mapped to each node.\n- **Dominant Cluster:** Shows which K-Means cluster controls that node (empty nodes in grey).\n- **Dominant Class:** Shows the most common actual future behaviour class in each node (SOM never saw these labels during training).")
    
    if not MINISOM_AVAILABLE:
        st.error("The `minisom` package is not installed. Please run `pip install minisom`.")
    else:
        grid_size = st.slider("Grid Size", 5, 15, 10)
        iterations = st.selectbox("Iterations", [5000, 10000], index=1)
        
        @st.cache_resource
        def get_cached_som(features, g_size, iters, seed=42):
            return train_som(features, grid_x=g_size, grid_y=g_size, iterations=iters, random_seed=seed)
            
        with st.spinner("Training Neural Network (cached)..."):
            som, X_scaled = get_cached_som(ml_features_unsup, grid_size, iterations)
            u_matrix, hit_map, bmu_coords = get_som_visualizations(som, X_scaled)
            
            # Print Quantization Error and Topographic Error
            q_error = som.quantization_error(X_scaled)
            t_error = som.topographic_error(X_scaled)
            st.write(f"**Quantization Error:** {q_error:.4f} | **Topographic Error:** {t_error:.4f}")
            
            # Get labels for clustering
            k_choice = st.session_state.get('seg_k', None)
            if k_choice is None:
                k_vals, _, sils = get_kmeans_metrics(ml_features_unsup, max_k=8)
                k_choice = k_vals[np.argmax(sils)]
            kmeans_labels = train_kmeans(ml_features_unsup, k=k_choice)
            
            sizes, _, profile = evaluate_clustering(raw_features, ml_features_unsup, kmeans_labels)
            interp = interpret_clusters(profile, sizes, len(raw_features), raw_features)
            kmeans_interp_labels = [interp[lbl] for lbl in kmeans_labels]
            
            # Plot
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(viz.plot_som_umatrix(u_matrix))
            with col2:
                st.pyplot(viz.plot_som_hitmap(hit_map))
                
            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(viz.plot_som_clusters(bmu_coords, kmeans_interp_labels, grid_size, title="Dominant Cluster per Node"))
            with col4:
                st.pyplot(viz.plot_som_clusters(bmu_coords, target_series['TargetClass'].values, grid_size, title="Dominant Future Class per Node"))

# --- Page: About ---
elif selection == "About Project":
    st.title("ℹ️ About the Project")
    st.markdown("""
    ### Problem Statement
    Using a customer's historical transaction behavior (Recency, Frequency, Monetary value), predict their value category (Inactive, Low/Medium, High) for the future using Decision Trees, Naive Bayes, and Support Vector Machines.
    
    ### Leakage Control
    - **Features** are computed strictly from data *before* `2011-09-01`.
    - **Target Class** is computed strictly from data *after* `2011-09-01`.
    - **Scaling** is fit entirely on the training set.
    
    ### Algorithms Used
    1. **Classification:** Decision Tree, Naive Bayes, SVM (Supervised, predicts future class).
    2. **Clustering:** K-Means, K-Medoids (Unsupervised, finds customer personas).
    3. **Association:** Apriori (Finds co-purchased products in shopping baskets).
    4. **Dimensionality Reduction:** PCA (Linear feature projection), SOM (Non-linear topological mapping).
    
    ### Limitations
    - **Moderate Accuracy:** Human purchasing behavior is noisy, resulting in moderate overall accuracy.
    - **Middle Class Hard to Predict:** The "Low/Medium" class is often misclassified as "Inactive".
    - **Single Dataset / UK-Only Apriori:** Models are trained on a single dataset, and Market Basket Analysis was restricted to the UK.
    - **Single Cutoff Date:** The prediction logic relies on a single fixed cutoff date rather than a rolling window.
    
    ### Future Work
    - Introduce deep learning models for sequence prediction (e.g., LSTMs for time-series purchases).
    - Deploy models via REST API for real-time inference.
    - Extend Apriori to include product categories instead of individual items.
    """)
