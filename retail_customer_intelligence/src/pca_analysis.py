import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(X, n_components=2):
    """
    Scales the input data, applies Principal Component Analysis (PCA) to reduce the feature space.
    Returns the transformed 2D data, the variance explained by each component, 
    and the cumulative variance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We initialize PCA. We default to 2 components for 2D plotting in the UI.
    # However, if n_components is None, it calculates for all features to show full variance.
    pca = PCA(n_components=n_components, random_state=42)
    
    # Fit and transform
    X_pca = pca.fit_transform(X_scaled)
    
    # Get variance metrics
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Convert the 2D projection into a DataFrame for easy plotting later
    if n_components == 2:
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    else:
        df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        
    return df_pca, explained_variance, cumulative_variance, pca
