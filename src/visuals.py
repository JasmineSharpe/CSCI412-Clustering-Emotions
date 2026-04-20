from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA



def project_with_pca(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca



def make_scatter_figure(df_with_coords: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df_with_coords["pc1"],
        df_with_coords["pc2"],
        c=df_with_coords["cluster"],
    )
    ax.set_title("Mood-Based Song Clusters (PCA Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    fig.tight_layout()
    return fig
