# src/data_pipeline/live_data_loader.py

"""
Live Open-Meteo data loader for operational frost prediction.

Retrieves the hourly Open-Meteo forecast, validates it, selects
tomorrow's local calendar day, and hands that day to
``data_pipeline.feature_mapper`` for translation into the trained
model's feature schema plus operational metadata. Kept separate from
feature_mapper because the model was trained on GeoSphere Austria
station observations while operational prediction runs on Open-Meteo
forecast variables.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.data_pipeline.feature_mapper import (
    build_model_features,
    build_radiation_features,
)

API_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

LATITUDE = 47.08
LONGITUDE = 15.44

LOCATION_NAME = "Graz, Austria"

TIMEZONE = "Europe/Vienna"

REQUEST_TIMEOUT_SECONDS = 30


HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "precipitation",
    "wind_gusts_10m",
    "wind_speed_10m",
    "soil_temperature_0cm",
]

REQUIRED_COLUMNS = [
    "time",
    *HOURLY_VARIABLES,
]


class LiveWeatherDataError(
    RuntimeError
):
    """
    Raised when live forecast data cannot be retrieved,
    validated, or transformed.
    """


def _validate_hourly_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Validate that all Open-Meteo variables required by the operational
    feature pipeline are available.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise LiveWeatherDataError(
            "Open-Meteo response is missing required "
            "hourly variables: "
            + ", ".join(missing_columns)
        )

    if df.empty:

        raise LiveWeatherDataError(
            "Open-Meteo returned an empty hourly forecast."
        )


def load_live_weather() -> pd.DataFrame:
    """
    Retrieve hourly Open-Meteo forecast data.

    Two forecast days are requested because the operational model
    evaluates tomorrow rather than the current calendar day.

    Explicit units:

    temperature: °C
    wind speed: km/h
    precipitation: mm

    Unit conversion required for model compatibility is performed
    later by feature_mapper.py.
    """

    params = {

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "hourly":
            ",".join(
                HOURLY_VARIABLES
            ),

        "forecast_days":
            2,

        "timezone":
            TIMEZONE,

        "temperature_unit":
            "celsius",

        "wind_speed_unit":
            "kmh",

        "precipitation_unit":
            "mm",
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        raise LiveWeatherDataError(
            "Unable to retrieve live weather data "
            f"from Open-Meteo: {exc}"
        ) from exc

    try:

        payload: dict[str, Any] = (
            response.json()
        )

    except ValueError as exc:

        raise LiveWeatherDataError(
            "Open-Meteo returned an invalid JSON response."
        ) from exc

    if "hourly" not in payload:

        reason = payload.get(
            "reason",
            "No hourly forecast was returned.",
        )

        raise LiveWeatherDataError(
            "Invalid Open-Meteo response: "
            f"{reason}"
        )

    hourly = pd.DataFrame(
        payload["hourly"]
    )

    _validate_hourly_dataframe(
        hourly
    )

    hourly["time"] = pd.to_datetime(
        hourly["time"],
        errors="coerce",
    )

    if hourly["time"].isna().any():

        raise LiveWeatherDataError(
            "One or more Open-Meteo timestamps "
            "could not be parsed."
        )

    for column in HOURLY_VARIABLES:

        hourly[column] = pd.to_numeric(
            hourly[column],
            errors="coerce",
        )

    missing_values = (
        hourly[
            HOURLY_VARIABLES
        ]
        .isna()
        .sum()
    )

    invalid_columns = (
        missing_values[
            missing_values > 0
        ]
    )

    if not invalid_columns.empty:

        details = ", ".join(
            f"{column}={count}"
            for column, count
            in invalid_columns.items()
        )

        raise LiveWeatherDataError(
            "Open-Meteo returned missing or "
            "non-numeric values: "
            + details
        )

    return hourly


def _select_prediction_day(
    hourly: pd.DataFrame,
    prediction_date: str,
) -> pd.DataFrame:
    """
    Select exactly the hourly forecast records belonging to the
    requested local prediction date.
    """

    selected = hourly[
        hourly["time"]
        .dt.strftime("%Y-%m-%d")
        == prediction_date
    ].copy()

    if selected.empty:

        raise LiveWeatherDataError(
            "No hourly forecast was found for "
            f"{prediction_date}."
        )

    if len(selected) != 24:

        raise LiveWeatherDataError(
            "Expected 24 hourly records for "
            f"{prediction_date}, but received "
            f"{len(selected)}."
        )

    return selected.reset_index(
        drop=True
    )


def build_feature_row() -> dict[str, Any]:
    """
    Build the complete operational feature row for tomorrow.

    The returned dictionary contains:

    - Random Forest model features
    - RadiationFrostAgent features
    - forecast metadata
    - feature-mapping metadata

    The GeoSphere/Open-Meteo transformations themselves are performed
    by feature_mapper.py.
    """

    hourly = load_live_weather()

    now_local = (
        datetime.now()
        .astimezone()
    )

    prediction_date = (
        now_local.date()
        + timedelta(days=1)
    ).isoformat()

    forecast = _select_prediction_day(
        hourly=hourly,
        prediction_date=prediction_date,
    )

    try:

        model_features = (
            build_model_features(
                forecast
            )
        )

        radiation_features = (
            build_radiation_features(
                forecast
            )
        )

    except (KeyError, ValueError) as exc:

        raise LiveWeatherDataError(
            "Unable to transform Open-Meteo forecast "
            f"into operational model features: {exc}"
        ) from exc

    row: dict[str, Any] = {}

    row.update(
        model_features
    )

    row.update(
        radiation_features
    )

    row.update(
        {
            "location":
                LOCATION_NAME,

            "latitude":
                LATITUDE,

            "longitude":
                LONGITUDE,

            "timezone":
                TIMEZONE,

            "forecast_generated":
                now_local.strftime(
                    "%Y-%m-%d %H:%M %Z"
                ),

            "prediction_date":
                prediction_date,

            "forecast_window_start":
                forecast[
                    "time"
                ].min().isoformat(),

            "forecast_window_end":
                forecast[
                    "time"
                ].max().isoformat(),

            "hourly_records_used":
                int(
                    len(forecast)
                ),

            "data_source":
                "Open-Meteo Forecast API",

            "training_data_source":
                (
                    "GeoSphere Austria "
                    "(Graz Universität Station)"
                ),

            "model_type":
                (
                    "Random Forest Frost "
                    "Prediction Model"
                ),

            "feature_schema":
                (
                    "GeoSphere-trained model schema "
                    "with explicit Open-Meteo "
                    "operational mappings"
                ),

            "feature_mapping_contains_proxies":
                True,
        }
    )

    return row


if __name__ == "__main__":

    try:

        feature_row = (
            build_feature_row()
        )

        print(feature_row)

    except LiveWeatherDataError as exc:

        print(f"Live weather data error: {exc}")

        raise SystemExit(
            1
        ) from exc