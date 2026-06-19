# src/test/test_synthetic_scenarios.py

from copy import deepcopy

import joblib

from src.agents.weather_agent import (
    WeatherAgent
)

from src.agents.radiation_frost_agent import (
    RadiationFrostAgent
)


def probability_to_risk(probability):

    probability = probability * 100

    if probability < 20:
        return "LOW"

    elif probability < 60:
        return "MEDIUM"

    else:
        return "HIGH"


def evaluate_scenario(
    title,
    sample_data,
    weather_agent,
    radiation_agent
):

    weather_result = (
        weather_agent.predict(
            sample_data
        )
    )

    radiation_result = (
        radiation_agent.assess(
            sample_data
        )
    )

    rf_risk = (
        probability_to_risk(
            weather_result[
                "frost_probability"
            ]
        )
    )

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)

    print(
        f"Temperature: "
        f"{radiation_result['temperature']:.1f} °C"
    )

    print(
        f"Wind Speed: "
        f"{radiation_result['wind_speed']:.1f}"
    )

    print(
        f"Cloud Cover: "
        f"{radiation_result['cloud_cover']:.1f}%"
    )

    print()

    print(
        f"Random Forest Risk: "
        f"{rf_risk}"
    )

    print(
        f"Radiation Frost Risk: "
        f"{radiation_result['risk_level']}"
    )


def main():

    weather_agent = WeatherAgent()

    radiation_agent = (
        RadiationFrostAgent()
    )

    base = joblib.load(
        "models/sample_row_frost.pkl"
    )

    # Scenario 1
    scenario_1 = deepcopy(base)

    scenario_1["temp_min"] = -1
    scenario_1["max_wind_gust"] = 15
    scenario_1["cloud_afternoon"] = 80

    # Scenario 2
    scenario_2 = deepcopy(base)

    scenario_2["temp_min"] = 0
    scenario_2["max_wind_gust"] = 2
    scenario_2["cloud_afternoon"] = 5

    # Scenario 3
    scenario_3 = deepcopy(base)

    scenario_3["temp_min"] = 1
    scenario_3["max_wind_gust"] = 3
    scenario_3["cloud_afternoon"] = 10

    evaluate_scenario(
        "SCENARIO 1 - COLD BUT CLOUDY/WINDY",
        scenario_1,
        weather_agent,
        radiation_agent
    )

    evaluate_scenario(
        "SCENARIO 2 - IDEAL RADIATION FROST",
        scenario_2,
        weather_agent,
        radiation_agent
    )

    evaluate_scenario(
        "SCENARIO 3 - BORDERLINE CASE",
        scenario_3,
        weather_agent,
        radiation_agent
    )


if __name__ == "__main__":

    main()