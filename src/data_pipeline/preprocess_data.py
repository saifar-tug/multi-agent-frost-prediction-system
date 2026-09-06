# src/data_pipeline/preprocess_data.py

import pandas as pd

from src.data_pipeline.feature_mapper import MODEL_FEATURES

# Single source of truth for the trained model's feature schema lives
# in feature_mapper.MODEL_FEATURES; kept under this name here since
# it's the name training/retraining code expects.
FEATURE_COLUMNS = MODEL_FEATURES


RENAME_MAP = {
    "reif": "frost",

    "tlmin": "temp_min",
    "tlmax": "temp_max",
    "tl_mittel": "temp_mean",

    "tsmin": "near_ground_temp_min",

    "rf_mittel": "humidity_mean",

    "dampf_mittel": "vapor_pressure_mean",

    "p_mittel": "pressure_mean",

    "bewm_i": "cloud_morning",
    "bewm_ii": "cloud_afternoon",

    "rr": "precipitation",

    "sicht_i": "visibility_morning",
    "sicht_ii": "visibility_afternoon",

    "tau": "dew",
    "nebel": "fog",

    "bft6": "wind_bft6",
    "bft8": "wind_bft8",

    "ffx": "max_wind_gust"
}


def load_and_preprocess(csv_path):

    df = pd.read_csv(csv_path)

    df = df.rename(
        columns=RENAME_MAP
    )

    df = df.dropna(
        subset=["frost"]
    )

    X = df[FEATURE_COLUMNS].copy()

    y = df["frost"].astype(int)

    X = X.fillna(
        X.median(numeric_only=True)
    )

    return X, y