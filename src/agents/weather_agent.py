# src/agents/weather_agent.py

import joblib
import pandas as pd


class WeatherAgent:

    def __init__(self):

        self.model = joblib.load(
            "models/random_forest_frost_model.pkl"
        )

        self.feature_columns = joblib.load(
            "models/feature_columns.pkl"
        )

    def predict(self, weather_data):

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in weather_data
        ]

        if missing_features:

            raise ValueError(
                "WeatherAgent received incomplete model input. "
                f"Missing features: {missing_features}"
            )

        input_df = pd.DataFrame(
            [weather_data]
        )

        input_df = input_df[
            self.feature_columns
        ]

        frost_probability = (
            self.model.predict_proba(
                input_df
            )[0][1]
        )

        frost_probability = float(
            frost_probability
        )

        frost_prediction = int(
            frost_probability >= 0.5
        )

        return {
            "agent": "WeatherAgent",

            "model_name":
                "Random Forest Frost Prediction Model",

            "training_data_source":
                "GeoSphere Austria",

            "operational_data_source":
                weather_data.get(
                    "data_source",
                    "Unknown"
                ),

            "location":
                weather_data.get(
                    "location",
                    "Unknown"
                ),

            "forecast_generated":
                weather_data.get(
                    "forecast_generated"
                ),

            "prediction_date":
                weather_data.get(
                    "prediction_date"
                ),

            "frost_probability":
                round(
                    frost_probability,
                    3
                ),

            "frost_prediction":
                frost_prediction,

            "prediction_label":
                (
                    "FROST"
                    if frost_prediction == 1
                    else "NO FROST"
                )
        }