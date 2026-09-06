# src/agents/soil_agent.py

class SoilAgent:

    def assess(self, weather_data):

        soil_temp = weather_data.get(
            "radiation_soil_temp_min"
        )

        if soil_temp is None:

            risk = "unknown"

            explanation = (
                "Open-Meteo soil-surface temperature data "
                "is not available for this forecast."
            )

        elif soil_temp <= 0:

            risk = "high"

            explanation = (
                "The forecast minimum soil-surface temperature "
                "at 0 cm is at or below freezing."
            )

        elif soil_temp <= 2:

            risk = "medium"

            explanation = (
                "The forecast minimum soil-surface temperature "
                "at 0 cm is close to freezing."
            )

        else:

            risk = "low"

            explanation = (
                "The forecast minimum soil-surface temperature "
                "at 0 cm remains above freezing."
            )

        return {
            "agent": "SoilAgent",

            "soil_temperature": soil_temp,

            "soil_risk": risk,

            "explanation": explanation
        }