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

        input_df = pd.DataFrame([weather_data])

        input_df = input_df[self.feature_columns]

        frost_probability = (
            self.model.predict_proba(input_df)[0][1]
        )
        '''
        Every time the pipeline runs, the Random Forest predicts again.

        But because the input row is the same each time, the output is also the same:
        e.g., frost_probability = 0.843 and frost_prediction = 1
        This is expected, since the model is deterministic given the same input.
        The LLM Orchestrator will receive the same weather output each time, but it can still generate different reports based on the combined outputs from all agents.
        '''

        frost_prediction = int(
            frost_probability >= 0.5
        )

        return {
            "agent": "WeatherAgent",
            "frost_probability": round(
                float(frost_probability), 3
            ),
            "frost_prediction": frost_prediction
        }