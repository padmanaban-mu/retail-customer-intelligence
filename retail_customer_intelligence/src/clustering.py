import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

def get_kmeans_metrics(X, max_k=8):
    """
    Scans K from 2 to max_k.
    Scales data internally.
    Returns lists of inertias (for elbow method) and silhouette scores.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    k_values = list(range(2, max_k + 1))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        
    return k_values, inertias, silhouette_scores

def train_kmeans(X, k=3):
    """
    Trains K-Means and returns cluster labels.
    Scales data internally.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans.fit_predict(X_scaled)

def train_kmedoids(X, k=3, max_iter=100):
    """
    Custom PAM-style K-Medoids implementation using a precomputed distance matrix.
    Scales data internally.
    Safe fallback to avoid scikit-learn-extra installation issues.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Precompute pairwise distances
    D = pairwise_distances(X_scaled, metric='euclidean')
    n = D.shape[0]
    
    # Initialize medoids randomly
    np.random.seed(42)
    medoids = np.random.choice(n, k, replace=False)
    
    # Assign points to nearest medoid
    labels = np.argmin(D[:, medoids], axis=1)
    cost = np.sum(np.min(D[:, medoids], axis=1))
    
    for _ in range(max_iter):
        best_medoids = medoids.copy()
        best_cost = cost
        
        # Try swapping each medoid with each non-medoid
        for i in range(k):
            for j in range(n):
                if j not in medoids:
                    temp_medoids = medoids.copy()
                    temp_medoids[i] = j
                    temp_cost = np.sum(np.min(D[:, temp_medoids], axis=1))
                    
                    if temp_cost < best_cost:
                        best_cost = temp_cost
                        best_medoids = temp_medoids
                        
        if best_cost == cost:
            break # Convergence
            
        cost = best_cost
        medoids = best_medoids
        
    labels = np.argmin(D[:, medoids], axis=1)
    return labels

def evaluate_clustering(X_original, X_model, labels):
    """
    Generates a profile table based on the original unscaled features for interpretability.
    Scales X_model internally for silhouette score.
    Returns cluster sizes, silhouette score, and the profile table.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_model)
    
    df_profile = X_original.copy()
    df_profile['Cluster'] = labels
    
    # Compute cluster sizes
    sizes = df_profile['Cluster'].value_counts().sort_index().to_dict()
    
    # Compute silhouette score
    sil_score = silhouette_score(X_scaled, labels)
    
    # Build Profile Table (mean of each feature per cluster)
    profile_table = df_profile.groupby('Cluster').mean()
    
    return sizes, sil_score, profile_table

def interpret_clusters(profile_table, sizes, total_customers, raw_features=None):
    """
    Derives rule-based interpretation labels derived from cluster statistics vs overall median.
    Make sure no cluster has fewer than ~1% of customers.
    """
    interpretations = {}
    
    # Calculate overall medians
    if raw_features is None:
        raw_features = profile_table
    r_med = raw_features['Recency'].median()
    f_med = raw_features['Frequency'].median()
    m_med = raw_features['Monetary'].median()
    
    for cluster_id, row in profile_table.iterrows():
        pct = sizes[cluster_id] / total_customers
        if pct < 0.01:
            interpretations[cluster_id] = "Outliers (Tiny Cluster)"
            continue
            
        r_tag = "Low Recency" if row['Recency'] <= r_med else "High Recency"
        f_tag = "High Frequency" if row['Frequency'] > f_med else "Low Frequency"
        m_tag = "High Monetary" if row['Monetary'] > m_med else "Low Monetary"
        
        # Build logic
        if r_tag == "Low Recency" and f_tag == "High Frequency" and m_tag == "High Monetary":
            label = "Champions: recent, frequent, high spend"
        elif r_tag == "Low Recency" and f_tag == "Low Frequency" and m_tag == "Low Monetary":
            label = "New / recent low-spend customers"
        elif r_tag == "High Recency" and f_tag == "Low Frequency" and m_tag == "Low Monetary":
            label = "Dormant low-value customers"
        elif r_tag == "High Recency" and m_tag == "High Monetary":
            label = "At-risk big spenders"
        else:
            label = "Regular / average customers"
            
        # Append unique suffix if label already used
        base_label = label
        suffix = 1
        while label in interpretations.values():
            label = f"{base_label} ({suffix})"
            suffix += 1
            
        interpretations[cluster_id] = label
            
    return interpretations
