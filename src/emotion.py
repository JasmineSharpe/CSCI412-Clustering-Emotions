from __future__ import annotations

import pandas as pd



def label_cluster_from_centroid(centroid_row: pd.Series) -> str:
    valence = centroid_row["valence"]
    energy = centroid_row["energy"]

    if valence >= 0.60 and energy >= 0.60:
        return "Excited / Joyful"
    if valence >= 0.60 and energy < 0.60:
        return "Calm / Content"
    if valence < 0.60 and energy >= 0.60:
        return "Tense / Intense"
    return "Melancholic / Reflective"



def build_cluster_summary(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    summary = df.groupby("cluster")[feature_columns].mean().round(3)
    summary["emotion_label"] = summary.apply(label_cluster_from_centroid, axis=1)
    summary["song_count"] = df.groupby("cluster").size()
    return summary.reset_index()
