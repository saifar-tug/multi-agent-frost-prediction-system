# src/agents/radiation_frost_agent.py
class RadiationFrostAgent:

    def __init__(self):

        self.agent_name = (
            "RadiationFrostAgent"
        )

    def assess(
        self,
        weather_data
    ):

        # Temperature

        temperature = weather_data.get(
            "radiation_temp_min",
            weather_data.get(
                "temp_min",
                999
            )
        )

        # Wind

        wind_speed = weather_data.get(
            "radiation_wind_speed",
            weather_data.get(
                "max_wind_gust",
                999
            )
        )

        # Cloud Cover

        cloud_cover = weather_data.get(
            "radiation_cloud_cover",
            weather_data.get(
                "cloud_afternoon",
                100
            )
        )

        if (
            temperature <= 0
            and wind_speed < 5
            and cloud_cover < 30
        ):

            risk_level = "HIGH"

        elif (
            temperature <= 2
            and wind_speed < 10
            and cloud_cover < 50
        ):

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        explanation = f"""
Radiation frost develops during clear and calm nights when heat escapes from the ground into the atmosphere.

Forecast minimum temperature: {temperature:.1f} °C
Average wind speed: {wind_speed:.1f} km/h
Average cloud cover: {cloud_cover:.1f} %

Risk Assessment: {risk_level}
"""

        return {

            "agent":
                self.agent_name,

            "data_source":
                weather_data.get(
                    "data_source",
                    "GeoSphere Austria"
                ),

            "location":
                weather_data.get(
                    "location",
                    "Graz, Austria"
                ),

            "forecast_generated":
                weather_data.get(
                    "forecast_generated",
                    "Historical Observation"
                ),

            "prediction_date":
                weather_data.get(
                    "prediction_date",
                    "Historical Record"
                ),

            "temperature":
                temperature,

            "wind_speed":
                wind_speed,

            "cloud_cover":
                cloud_cover,

            "risk_level":
                risk_level,

            "explanation":
                explanation
        }