from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS


REQUIRED_COLUMNS = {
    "track_name",
    "artist",
    "track_genre",
    "popularity",
    *FEATURE_COLUMNS,
}


def load_dataset(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(sorted(missing))
        )
    return df.copy()



def prepare_features(df: pd.DataFrame):
    clean_df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df[FEATURE_COLUMNS])
    return clean_df, X_scaled, scaler
