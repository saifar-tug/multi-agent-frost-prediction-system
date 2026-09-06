# src/agents/radiation_frost_agent.py


class RadiationFrostAgent:

    AGENT_NAME = "RadiationFrostAgent"

    def assess(self, weather_data):

        air_temperature = weather_data.get(
            "radiation_temp_min"
        )

        soil_temperature = weather_data.get(
            "radiation_soil_temp_min"
        )

        wind_speed = weather_data.get(
            "radiation_wind_speed"
        )

        cloud_cover = weather_data.get(
            "radiation_cloud_cover"
        )

        required_values = {
            "air_temperature": air_temperature,
            "wind_speed": wind_speed,
            "cloud_cover": cloud_cover
        }

        missing_values = [
            name
            for name, value in required_values.items()
            if value is None
        ]

        if missing_values:

            risk_level = "UNKNOWN"

            explanation = (
                "Radiation-frost risk could not be fully assessed "
                "because required forecast variables are missing: "
                + ", ".join(missing_values)
                + "."
            )

        else:

            # Radiation frost is favored by:
            #
            # - low nighttime air temperature
            # - weak wind
            # - low cloud cover
            #
            # Soil-surface temperature is included as supporting
            # ground-level evidence but is not required for the
            # primary classification.

            if (
                air_temperature <= 0
                and wind_speed < 5
                and cloud_cover < 30
            ):

                risk_level = "HIGH"

            elif (
                air_temperature <= 2
                and wind_speed < 10
                and cloud_cover < 50
            ):

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"

            explanation_lines = [

                "Radiation frost is most likely during cold, "
                "clear, and calm conditions because the ground "
                "can lose heat rapidly through longwave radiation.",

                "",

                f"Forecast minimum 2 m air temperature: "
                f"{air_temperature:.1f} °C",

                f"Average wind speed: "
                f"{wind_speed:.1f} km/h",

                f"Average cloud cover: "
                f"{cloud_cover:.1f} %"
            ]

            if soil_temperature is not None:

                explanation_lines.extend([
                    "",
                    f"Forecast minimum soil-surface temperature "
                    f"(0 cm): {soil_temperature:.1f} °C"
                ])

            explanation_lines.extend([
                "",
                f"Radiation Frost Risk: {risk_level}"
            ])

            explanation = "\n".join(
                explanation_lines
            )

        return {

            "agent":
                self.AGENT_NAME,

            "data_source":
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
                    "forecast_generated",
                    "N/A"
                ),

            "prediction_date":
                weather_data.get(
                    "prediction_date",
                    "N/A"
                ),

            "forecast_window_start":
                weather_data.get(
                    "forecast_window_start"
                ),

            "forecast_window_end":
                weather_data.get(
                    "forecast_window_end"
                ),

            "air_temperature":
                air_temperature,

            "soil_surface_temperature":
                soil_temperature,

            "soil_temperature_depth_cm":
                0 if soil_temperature is not None else None,

            "wind_speed":
                wind_speed,

            "cloud_cover":
                cloud_cover,

            "risk_level":
                risk_level,

            "missing_inputs":
                missing_values,

            "explanation":
                explanation
        }