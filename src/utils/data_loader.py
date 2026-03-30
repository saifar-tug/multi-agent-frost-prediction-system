import pandas as pd


def load_graz_frost_dataset(path: str):
    """
    Loads ECA&D minimum temperature data for Graz (GRAZ-UNIVERSITAET).
    
    Processing steps:
    - Skips metadata section
    - Filters years 1990–2024
    - Removes missing and suspect values
    - Converts temperature from tenths °C to °C
    - Creates frost label (Tmin <= 0°C)
    - Creates 1-day-ahead forecasting target
    
    Returns:
        DataFrame with:
        - DATE
        - TN_C (temperature on day t)
        - target_frost (frost on day t+1)
    """

    #Locate header line (skip metadata block)
    with open(path, "r") as f:
        lines = f.readlines()

    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("STAID"):
            header_index = i
            break

    if header_index is None:
        raise ValueError("Header not found in dataset.")

    #Load data section
    df = pd.read_csv(
        path,
        skiprows=header_index,
        sep=",",
        engine="python"
    )

    df.columns = [c.strip() for c in df.columns]

    #Convert and filter date
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")

    df = df[
        (df["DATE"].dt.year >= 1990) &
        (df["DATE"].dt.year <= 2024)
    ]

    #missing and invalid data
    df = df[df["TN"] != -9999]
    df = df[df["Q_TN"] == 0]

    #Convert temperature
    df["TN_C"] = df["TN"] / 10.0

    #Create same-day frost label
    df["frost"] = (df["TN_C"] <= 0).astype(int)

    #Create 1-day-ahead forecasting target
    df["target_frost"] = df["frost"].shift(-1)

    #Remove last row (no next-day label)
    df = df.dropna().reset_index(drop=True)

    #Return forecasting dataset
    df = df[["DATE", "TN_C", "target_frost"]]

    return df