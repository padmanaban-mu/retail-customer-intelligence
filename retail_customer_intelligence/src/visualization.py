import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def plot_revenue_over_time(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    revenue_daily = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum()
    ax.plot(revenue_daily.index, revenue_daily.values, color='teal')
    ax.set_title("Daily Revenue")
    ax.set_ylabel("Revenue (£)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_model_comparison(metrics):
    models = list(metrics.keys())
    accuracies = [metrics[m]['accuracy'] for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=models, y=accuracies, ax=ax, palette='viridis')
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
    plt.tight_layout()
    return fig

def plot_confusion_matrices(metrics, class_names=None):
    # Only plot for the 3 actual models, not baseline
    models = [m for m in metrics.keys() if 'Baseline' not in m]
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4))
    
    for i, model in enumerate(models):
        cm = metrics[model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f"{model} Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        
    plt.tight_layout()
    return fig

def plot_elbow_silhouette(k_values, inertias, sil_scores):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(k_values, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_values, sil_scores, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title("K-Means Selection Metrics")
    return fig

def plot_pca_scatter(df_pca, labels=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue=labels, palette='tab10', alpha=0.7, ax=ax)
    else:
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', alpha=0.5, ax=ax)
    ax.set_title("2D PCA Projection")
    plt.tight_layout()
    return fig

def plot_som_umatrix(u_matrix):
    fig, ax = plt.subplots(figsize=(8, 8))
    # Display the U-matrix
    cax = ax.pcolor(u_matrix.T, cmap='bone_r')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("SOM U-Matrix (Distance Map)")
    plt.tight_layout()
    return fig

def plot_som_hitmap(hit_map):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(hit_map.T, annot=True, fmt='g', cmap='YlGnBu', ax=ax)
    ax.set_title("SOM Hit Map (Customer Density)")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_som_clusters(bmu_coords, labels, grid_size, title="Dominant Cluster per Node"):
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from collections import defaultdict

    unique_labels = sorted(list(set(labels)))
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    
    # Initialize grid with NaN
    cluster_grid = np.full((grid_size, grid_size), np.nan)
    
    # Collect labels for each cell
    cell_labels = defaultdict(list)
    for coord, label in zip(bmu_coords, labels):
        cell_labels[coord].append(label)
        
    # Find majority label per cell
    for (x, y), lbls in cell_labels.items():
        if len(lbls) > 0:
            majority_label = max(set(lbls), key=lbls.count)
            cluster_grid[x, y] = label_to_id[majority_label]
            
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [plt.get_cmap('tab10')(i % 10) for i in range(len(unique_labels))]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(colors))
    
    sns.heatmap(cluster_grid.T, cmap=cmap, norm=norm, ax=ax, cbar=False, linewidths=0.5, linecolor='gray')
    ax.set_title(title)
    ax.invert_yaxis()
    
    patches = [mpatches.Patch(color=colors[i], label=str(lbl)) for i, lbl in enumerate(unique_labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_before_after_histogram(df_before, df_after, column='Monetary'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.histplot(df_before[column], bins=50, ax=axes[0], color='lightcoral')
    axes[0].set_title(f"Before: {column} (Raw)")
    
    sns.histplot(df_after[column], bins=50, ax=axes[1], color='lightgreen')
    axes[1].set_title(f"After: {column} (Clipped & Log1p)")
    
    plt.tight_layout()
    return fig
