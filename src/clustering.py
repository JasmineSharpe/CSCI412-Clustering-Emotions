from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def find_best_k(X_scaled, k_min: int = 2, k_max: int = 7):
    results = []
    best_k = None
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        inertia = model.inertia_
        results.append({"k": k, "silhouette_score": score, "inertia": inertia})

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, results



def fit_kmeans(X_scaled, n_clusters: int):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels
