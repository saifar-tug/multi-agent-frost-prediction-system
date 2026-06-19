# src/data_pipeline/live_data_loader.py

from datetime import datetime
from datetime import timedelta

import requests
import pandas as pd


LATITUDE = 47.08
LONGITUDE = 15.44

LOCATION_NAME = "Graz, Austria"


def load_live_weather():

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude": LATITUDE,

        "longitude": LONGITUDE,

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "surface_pressure",
            "cloud_cover",
            "visibility",
            "precipitation",
            "wind_gusts_10m",
            "wind_speed_10m"
        ]),

        "forecast_days": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    hourly = pd.DataFrame(
        data["hourly"]
    )

    return hourly


def build_feature_row():

    df = load_live_weather()

    now = datetime.now()

    prediction_date = (
        now + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    generated_at = now.strftime(
        "%Y-%m-%d %H:%M"
    )

    row = {

        # -------------------------
        # Model Features
        # -------------------------

        "temp_min":
            df["temperature_2m"].min(),

        "temp_max":
            df["temperature_2m"].max(),

        "temp_mean":
            df["temperature_2m"].mean(),

        "soil_temp_min":
            df["temperature_2m"].min(),

        "humidity_mean":
            df["relative_humidity_2m"].mean(),

        "vapor_pressure_mean":
            df["dew_point_2m"].mean(),

        "pressure_mean":
            df["surface_pressure"].mean(),

        "cloud_morning":
            df["cloud_cover"][:12].mean(),

        "cloud_afternoon":
            df["cloud_cover"][12:].mean(),

        "precipitation":
            df["precipitation"].sum(),

        "visibility_morning":
            df["visibility"][:12].mean(),

        "visibility_afternoon":
            df["visibility"][12:].mean(),

        "dew":
            int(
                df["dew_point_2m"].mean() > 0
            ),

        "fog":
            int(
                df["visibility"].mean() < 1000
            ),

        "wind_bft6":
            int(
                df["wind_speed_10m"].max() >= 39
            ),

        "wind_bft8":
            int(
                df["wind_speed_10m"].max() >= 62
            ),

        "max_wind_gust":
            df["wind_gusts_10m"].max(),

        # -------------------------
        # Radiation Frost Features
        # -------------------------

        "radiation_temp_min":
            df["temperature_2m"].min(),

        "radiation_wind_speed":
            df["wind_speed_10m"].mean(),

        "radiation_wind_gust":
            df["wind_gusts_10m"].max(),

        "radiation_cloud_cover":
            df["cloud_cover"].mean(),

        # -------------------------
        # Metadata
        # -------------------------

        "location":
            LOCATION_NAME,

        "forecast_generated":
            generated_at,

        "prediction_date":
            prediction_date,

        "data_source":
            "Open-Meteo API"
    }

    return row


if __name__ == "__main__":

    feature_row = build_feature_row()

    print("\n" + "=" * 60)
    print("LIVE WEATHER DATA")
    print("=" * 60)

    print(
        f"Location: "
        f"{feature_row['location']}"
    )

    print(
        f"Data Source: "
        f"{feature_row['data_source']}"
    )

    print(
        f"Forecast Generated: "
        f"{feature_row['forecast_generated']}"
    )

    print(
        f"Prediction Date: "
        f"{feature_row['prediction_date']}"
    )

    print("\nMODEL FEATURES\n")

    excluded = {
        "location",
        "forecast_generated",
        "prediction_date",
        "data_source"
    }

    for key, value in feature_row.items():

        if key not in excluded:

            print(
                f"{key}: {value}"
            )