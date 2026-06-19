# src/data_pipeline/preprocess_data.py

import pandas as pd


FEATURE_COLUMNS = [
    "temp_min",
    "temp_max",
    "temp_mean",
    "soil_temp_min",
    "humidity_mean",
    "vapor_pressure_mean",
    "pressure_mean",
    "cloud_morning",
    "cloud_afternoon",
    "precipitation",
    "visibility_morning",
    "visibility_afternoon",
    "dew",
    "fog",
    "wind_bft6",
    "wind_bft8",
    "max_wind_gust"
]


RENAME_MAP = {
    "reif": "frost",

    "tlmin": "temp_min",
    "tlmax": "temp_max",
    "tl_mittel": "temp_mean",

    "tsmin": "soil_temp_min",

    "rf_mittel": "humidity_mean",

    "dampf_mittel": "vapor_pressure_mean",

    "p_mittel": "pressure_mean",

    "bewd_i": "cloud_morning",
    "bewd_ii": "cloud_afternoon",

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