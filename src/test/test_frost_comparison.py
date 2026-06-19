# src/test/test_frost_comparison.py

import joblib

from src.agents.weather_agent import (
    WeatherAgent
)

from src.agents.radiation_frost_agent import (
    RadiationFrostAgent
)

from src.data_pipeline.live_data_loader import (
    build_feature_row
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
    scenario_name,
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

    rf_probability = (
        weather_result[
            "frost_probability"
        ] * 100
    )

    rf_risk = (
        probability_to_risk(
            weather_result[
                "frost_probability"
            ]
        )
    )

    radiation_risk = (
        radiation_result[
            "risk_level"
        ]
    )

    agreement = (
        rf_risk == radiation_risk
    )

    print("\n" + "-" * 60)

    print(
        f"Scenario: {scenario_name}"
    )

    print("-" * 60)

    print("\nRandom Forest")

    print(
        f"Probability: "
        f"{rf_probability:.1f}%"
    )

    print(
        f"Risk Level: "
        f"{rf_risk}"
    )

    print("\nRadiation Frost")

    print(
        f"Risk Level: "
        f"{radiation_risk}"
    )

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

    print("\nAgreement Between Methods")

    print(
        "YES"
        if agreement
        else "NO"
    )

    return {

        "scenario":
            scenario_name,

        "rf_risk":
            rf_risk,

        "radiation_risk":
            radiation_risk,

        "agreement":
            agreement
    }


def main():

    weather_agent = WeatherAgent()

    radiation_agent = (
        RadiationFrostAgent()
    )

    print("\n" + "=" * 60)
    print("FROST DETECTION COMPARISON")
    print("=" * 60)

    frost_sample = joblib.load(
        "models/sample_row_frost.pkl"
    )

    recent_sample = joblib.load(
        "models/sample_row_recent.pkl"
    )

    live_sample = (
        build_feature_row()
    )

    results = []

    results.append(
        evaluate_scenario(
            "Historical Frost Event",
            frost_sample,
            weather_agent,
            radiation_agent
        )
    )

    results.append(
        evaluate_scenario(
            "Latest Historical Observation",
            recent_sample,
            weather_agent,
            radiation_agent
        )
    )

    results.append(
        evaluate_scenario(
            "Live Open-Meteo Forecast",
            live_sample,
            weather_agent,
            radiation_agent
        )
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in results:

        print(
            f"\nScenario: {result['scenario']}"
        )

        print(
            f"Random Forest Risk: "
            f"{result['rf_risk']}"
        )

        print(
            f"Radiation Frost Risk: "
            f"{result['radiation_risk']}"
        )

        print(
            f"Agreement: "
            f"{'YES' if result['agreement'] else 'NO'}"
        )


if __name__ == "__main__":

    main()