import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False

def train_som(X, grid_x=10, grid_y=10, sigma=1.0, learning_rate=0.5, iterations=10000, random_seed=42):
    """
    Scales data, trains a Self-Organizing Map (SOM) using the MiniSom library.
    Returns the trained SOM object and scaled X.
    """
    if not MINISOM_AVAILABLE:
        raise ImportError("The 'minisom' package is not installed. Please run `pip install minisom`.")
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    # Convert DataFrame to numpy array if necessary
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
        
    input_len = X_scaled.shape[1]
    
    som = MiniSom(grid_x, grid_y, input_len, sigma=sigma, learning_rate=learning_rate, random_seed=random_seed)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, iterations)
    
    return som, X_scaled

def get_som_visualizations(som, X_scaled):
    """
    Extracts the U-matrix (distance map), hit map (activation frequencies), 
    and the Best Matching Unit (BMU) coordinates for each customer.
    """
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
        
    u_matrix = som.distance_map()
    hit_map = som.activation_response(X_scaled)
    bmu_coords = [som.winner(x) for x in X_scaled]
    
    return u_matrix, hit_map, bmu_coords
