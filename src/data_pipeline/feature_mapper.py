# src/data_pipeline/feature_mapper.py

"""
Feature mapping for operational frost prediction.

The Random Forest model was trained on daily GeoSphere Austria station
variables, but operational prediction uses hourly Open-Meteo forecast
variables. This module translates Open-Meteo forecasts into the
model's feature schema, converting units and recording each mapping as
either a direct/derived equivalent or an explicit operational proxy.

Note: the model feature ``near_ground_temp_min`` corresponds to the
GeoSphere variable ``tsmin`` (minimum near-ground air temperature); the
Open-Meteo mapping used here is a proxy and must not be described as
measured soil temperature.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


MODEL_FEATURES = [
    "temp_min",
    "temp_max",
    "temp_mean",
    "near_ground_temp_min",
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
    "max_wind_gust",
]


FEATURE_MAPPING = {
    "temp_min": {
        "geosphere_variable": "tlmin",
        "operational_source": "temperature_2m",
        "mapping_type": "derived",
        "description": "Minimum 2 m air temperature.",
    },
    "temp_max": {
        "geosphere_variable": "tlmax",
        "operational_source": "temperature_2m",
        "mapping_type": "derived",
        "description": "Maximum 2 m air temperature.",
    },
    "temp_mean": {
        "geosphere_variable": "tl_mittel",
        "operational_source": "temperature_2m",
        "mapping_type": "derived",
        "description": "Mean 2 m air temperature.",
    },
    "near_ground_temp_min": {
        "geosphere_variable": "tsmin",
        "operational_source": "temperature_2m",
        "mapping_type": "proxy",
        "description": (
            "GeoSphere tsmin represents "
            "minimum near-ground air temperature. Open-Meteo "
            "temperature_2m is used as an operational proxy because "
            "the exact GeoSphere 5 cm air-temperature variable is not "
            "available in the current forecast request."
        ),
    },
    "humidity_mean": {
        "geosphere_variable": "rf_mittel",
        "operational_source": "relative_humidity_2m",
        "mapping_type": "derived",
        "description": "Mean relative humidity.",
    },
    "vapor_pressure_mean": {
        "geosphere_variable": "dampf_mittel",
        "operational_source": "dew_point_2m",
        "mapping_type": "derived",
        "description": (
            "Actual vapour pressure derived from forecast dew-point "
            "temperature using the Magnus approximation."
        ),
    },
    "pressure_mean": {
        "geosphere_variable": "p_mittel",
        "operational_source": "surface_pressure",
        "mapping_type": "derived",
        "description": "Mean surface/station pressure in hPa.",
    },
    "cloud_morning": {
        "geosphere_variable": "bewm_i",
        "operational_source": "cloud_cover",
        "mapping_type": "proxy",
        "description": (
            "Morning cloud-cover proxy derived from hourly forecast "
            "cloud cover. It is not an exact reproduction of the "
            "GeoSphere observation-time variable."
        ),
    },
    "cloud_afternoon": {
        "geosphere_variable": "bewm_ii",
        "operational_source": "cloud_cover",
        "mapping_type": "proxy",
        "description": (
            "Afternoon cloud-cover proxy derived from hourly forecast "
            "cloud cover. It is not an exact reproduction of the "
            "GeoSphere observation-time variable."
        ),
    },
    "precipitation": {
        "geosphere_variable": "rr",
        "operational_source": "precipitation",
        "mapping_type": "derived",
        "description": "Total precipitation over the forecast day.",
    },
    "visibility_morning": {
        "geosphere_variable": "sicht_i",
        "operational_source": "visibility",
        "mapping_type": "proxy",
        "description": (
            "Morning visibility proxy derived from hourly forecast "
            "visibility."
        ),
    },
    "visibility_afternoon": {
        "geosphere_variable": "sicht_ii",
        "operational_source": "visibility",
        "mapping_type": "proxy",
        "description": (
            "Afternoon visibility proxy derived from hourly forecast "
            "visibility."
        ),
    },
    "dew": {
        "geosphere_variable": "tau",
        "operational_source": (
            "relative_humidity_2m + temperature_2m + dew_point_2m"
        ),
        "mapping_type": "proxy",
        "description": (
            "Operational dew-event proxy based on humidity and "
            "temperature/dew-point spread."
        ),
    },
    "fog": {
        "geosphere_variable": "nebel",
        "operational_source": "visibility",
        "mapping_type": "proxy",
        "description": (
            "Operational fog-event proxy based on visibility below "
            "1,000 metres."
        ),
    },
    "wind_bft6": {
        "geosphere_variable": "bft6",
        "operational_source": "wind_speed_10m",
        "mapping_type": "proxy",
        "description": (
            "Operational Beaufort-6 indicator derived from maximum "
            "forecast wind speed."
        ),
    },
    "wind_bft8": {
        "geosphere_variable": "bft8",
        "operational_source": "wind_speed_10m",
        "mapping_type": "proxy",
        "description": (
            "Operational Beaufort-8 indicator derived from maximum "
            "forecast wind speed."
        ),
    },
    "max_wind_gust": {
        "geosphere_variable": "ffx",
        "operational_source": "wind_gusts_10m",
        "mapping_type": "derived",
        "description": (
            "Maximum forecast wind gust converted from km/h to m/s "
            "for compatibility with the GeoSphere training variable."
        ),
    },
}


def mean_vapor_pressure_hpa(
    dew_point_celsius: pd.Series,
) -> float:
    """
    Calculate mean actual vapour pressure from dew-point temperature.

    Magnus approximation:

        e = 6.112 * exp((17.67 * Td) / (Td + 243.5))

    where:
        Td = dew-point temperature in °C
        e  = vapour pressure in hPa
    """

    exponent = (
        17.67 * dew_point_celsius
        / (dew_point_celsius + 243.5)
    )

    vapor_pressure = (
        6.112 * exponent.apply("exp")
    )

    return float(
        vapor_pressure.mean()
    )


def derive_dew_indicator(
    forecast: pd.DataFrame,
) -> int:
    """
    Create an operational proxy for the GeoSphere dew-event variable.

    Dew-favouring conditions are considered present when at least one
    hourly forecast satisfies:

    - relative humidity >= 90 %
    - temperature - dew point <= 2 °C

    This is a proxy, not a direct observation of dew.
    """

    temperature_dew_spread = (
        forecast["temperature_2m"]
        - forecast["dew_point_2m"]
    )

    dew_conditions = (
        (
            forecast["relative_humidity_2m"]
            >= 90.0
        )
        & (
            temperature_dew_spread
            <= 2.0
        )
    )

    return int(
        dew_conditions.any()
    )


def derive_fog_indicator(
    forecast: pd.DataFrame,
) -> int:
    """
    Create an operational proxy for a fog event.

    Fog is considered plausible if horizontal visibility falls below
    1,000 metres during at least one hourly forecast record.
    """

    return int(
        (
            forecast["visibility"]
            < 1000.0
        ).any()
    )


def kmh_to_mps(
    speed_kmh: float,
) -> float:
    """
    Convert kilometres per hour to metres per second.
    """

    return float(
        speed_kmh / 3.6
    )


def build_model_features(
    forecast: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Transform one complete forecast day into the 17 features expected
    by the trained Random Forest model.

    The function deliberately keeps proxy mappings visible rather than
    presenting them as exact GeoSphere/Open-Meteo equivalents.
    """

    if forecast.empty:
        raise ValueError(
            "Cannot build model features from an empty forecast."
        )

    morning = forecast[
        forecast["time"].dt.hour < 12
    ]

    afternoon = forecast[
        forecast["time"].dt.hour >= 12
    ]

    max_wind_speed_kmh = float(
        forecast["wind_speed_10m"].max()
    )

    max_wind_gust_kmh = float(
        forecast["wind_gusts_10m"].max()
    )

    features: dict[str, float | int] = {
        "temp_min": float(
            forecast["temperature_2m"].min()
        ),

        "temp_max": float(
            forecast["temperature_2m"].max()
        ),

        "temp_mean": float(
            forecast["temperature_2m"].mean()
        ),

        # IMPORTANT:
        #
        # The saved RF model expects the historical feature name
        # "near_ground_temp_min", corresponding to GeoSphere variable tsmin
        # represents near-ground AIR temperature.
        #
        # Open-Meteo temperature_2m is therefore used as an explicit
        # operational proxy. We must not describe this value as actual
        # soil temperature.
        "near_ground_temp_min": float(
            forecast["temperature_2m"].min()
        ),

        "humidity_mean": float(
            forecast[
                "relative_humidity_2m"
            ].mean()
        ),

        "vapor_pressure_mean":
            mean_vapor_pressure_hpa(
                forecast["dew_point_2m"]
            ),

        "pressure_mean": float(
            forecast[
                "surface_pressure"
            ].mean()
        ),

        # These remain operational proxies until the original
        # GeoSphere observation-time semantics are reproduced exactly.
        "cloud_morning": float(
            morning["cloud_cover"].mean()
        ),

        "cloud_afternoon": float(
            afternoon["cloud_cover"].mean()
        ),

        "precipitation": float(
            forecast[
                "precipitation"
            ].sum()
        ),

        "visibility_morning": float(
            morning["visibility"].mean()
        ),

        "visibility_afternoon": float(
            afternoon["visibility"].mean()
        ),

        "dew":
            derive_dew_indicator(
                forecast
            ),

        "fog":
            derive_fog_indicator(
                forecast
            ),

        "wind_bft6": int(
            max_wind_speed_kmh
            >= 39.0
        ),

        "wind_bft8": int(
            max_wind_speed_kmh
            >= 62.0
        ),

        # Open-Meteo is explicitly requested in km/h.
        # GeoSphere ffx was represented in m/s in the training data.
        "max_wind_gust":
            kmh_to_mps(
                max_wind_gust_kmh
            ),
    }

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            "Feature mapping failed. Missing features: "
            + ", ".join(missing_features)
        )

    return features


def build_radiation_features(
    forecast: pd.DataFrame,
) -> dict[str, float]:
    """
    Build meteorological features used by RadiationFrostAgent.

    Unlike the Random Forest feature ``near_ground_temp_min``,
    ``radiation_soil_temp_min`` genuinely represents Open-Meteo
    soil/surface temperature at 0 cm.
    """

    return {
        "radiation_temp_min": float(
            forecast[
                "temperature_2m"
            ].min()
        ),

        "radiation_soil_temp_min": float(
            forecast[
                "soil_temperature_0cm"
            ].min()
        ),

        "radiation_wind_speed": float(
            forecast[
                "wind_speed_10m"
            ].mean()
        ),

        "radiation_wind_gust": float(
            forecast[
                "wind_gusts_10m"
            ].max()
        ),

        "radiation_cloud_cover": float(
            forecast[
                "cloud_cover"
            ].mean()
        ),
    }


def get_model_dataframe(
    feature_row: dict[str, Any],
) -> pd.DataFrame:
    """
    Return the exact ordered feature dataframe expected by the
    Random Forest model.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in feature_row
    ]

    if missing_features:
        raise ValueError(
            "Feature row is missing Random Forest features: "
            + ", ".join(missing_features)
        )

    return pd.DataFrame(
        [
            {
                feature: feature_row[feature]
                for feature in MODEL_FEATURES
            }
        ],
        columns=MODEL_FEATURES,
    )
