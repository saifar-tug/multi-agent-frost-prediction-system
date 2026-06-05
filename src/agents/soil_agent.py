# src/agents/soil_agent.py
class SoilAgent:

    def assess(self, weather_data):

        soil_temp = weather_data.get("soil_temp_min")

        if soil_temp is None:

            risk = "unknown"

            explanation = (
                "No soil temperature available."
            )

        elif soil_temp <= 0:

            risk = "high"

            explanation = (
                "Minimum soil temperature is below freezing."
            )

        elif soil_temp <= 2:

            risk = "medium"

            explanation = (
                "Minimum soil temperature is close to freezing."
            )

        else:

            risk = "low"

            explanation = (
                "Minimum soil temperature is above freezing."
            )

        return {
            "agent": "SoilAgent",
            "soil_temperature": soil_temp,
            "soil_risk": risk,
            "explanation": explanation
        }