from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from src.clustering import find_best_k, fit_kmeans
from src.config import DISPLAY_COLUMNS, FEATURE_COLUMNS
from src.data_utils import load_dataset, prepare_features
from src.emotion import build_cluster_summary
from src.visuals import make_scatter_figure, project_with_pca

st.set_page_config(page_title="Music Mood Clustering", layout="wide")
st.title("Clustering Emotion: A Computational Exploration of Music and Mood")
st.write(
    "Upload a song dataset with Spotify-style audio features, then explore mood-based clusters using K-means."
)

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    use_sample = st.checkbox("Use included sample dataset", value=True)
    auto_k = st.checkbox("Choose best k automatically", value=True)
    manual_k = st.slider("Manual number of clusters", min_value=2, max_value=7, value=4)

file_to_use = None
if uploaded_file is not None:
    file_to_use = uploaded_file
elif use_sample:
    file_to_use = "data/sample_songs.csv"

if file_to_use is None:
    st.info("Upload a dataset or enable the sample dataset to begin.")
    st.stop()

try:
    df = load_dataset(file_to_use)
    clean_df, X_scaled, scaler = prepare_features(df)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

best_k, k_results = find_best_k(X_scaled)
k_results_df = pd.DataFrame(k_results)
selected_k = best_k if auto_k else manual_k
model, labels = fit_kmeans(X_scaled, selected_k)
clean_df["cluster"] = labels

coords, pca = project_with_pca(X_scaled)
clean_df["pc1"] = coords[:, 0]
clean_df["pc2"] = coords[:, 1]
summary_df = build_cluster_summary(clean_df, FEATURE_COLUMNS)

st.subheader("Dataset Preview")
st.dataframe(clean_df[DISPLAY_COLUMNS + FEATURE_COLUMNS].head(10), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Selection")
    st.write(f"Recommended number of clusters: **{best_k}**")
    st.write(f"Selected number of clusters: **{selected_k}**")
    st.dataframe(k_results_df, use_container_width=True)

with col2:
    st.subheader("PCA Cluster Visualization")
    fig = make_scatter_figure(clean_df)
    st.pyplot(fig)

st.subheader("Cluster Interpretation")
st.dataframe(summary_df, use_container_width=True)

selected_cluster = st.selectbox("Choose a cluster to inspect", sorted(clean_df["cluster"].unique().tolist()))
filtered = clean_df[clean_df["cluster"] == selected_cluster].copy()
filtered = filtered.sort_values(by=["popularity", "valence", "energy"], ascending=False)

st.subheader(f"Songs in Cluster {selected_cluster}")
st.dataframe(filtered[DISPLAY_COLUMNS + ["cluster"] + FEATURE_COLUMNS], use_container_width=True)

csv_buffer = io.StringIO()
clean_df.to_csv(csv_buffer, index=False)
st.download_button(
    "Download clustered results as CSV",
    data=csv_buffer.getvalue(),
    file_name="clustered_music_results.csv",
    mime="text/csv",
)
