from src.agents.radiation_frost_agent import (
    RadiationFrostAgent
)

from src.data_pipeline.live_data_loader import (
    build_feature_row
)


agent = RadiationFrostAgent()

weather_data = (
    build_feature_row()
)

result = (
    agent.assess(
        weather_data
    )
)

print("\n" + "=" * 60)
print("RADIATION FROST ASSESSMENT")
print("=" * 60)

print(
    f"Data Source: {result['data_source']}"
)

print(
    f"Location: {result['location']}"
)

print(
    f"Forecast Generated: "
    f"{result['forecast_generated']}"
)

print(
    f"Prediction Date: "
    f"{result['prediction_date']}"
)

print(
    f"Minimum Temperature: "
    f"{result['temperature']:.1f} °C"
)

print(
    f"Average Wind Speed: "
    f"{result['wind_speed']:.1f}"
)

print(
    f"Average Cloud Cover: "
    f"{result['cloud_cover']:.1f} %"
)

print(
    f"Radiation Frost Risk: "
    f"{result['risk_level']}"
)

print("\nExplanation:\n")

print(
    result["explanation"]
)