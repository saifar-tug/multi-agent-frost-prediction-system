# src/test/test_frost_comparison.py

import joblib
import pytest

from src.agents.weather_agent import WeatherAgent
from src.agents.radiation_frost_agent import RadiationFrostAgent

from src.data_pipeline.live_data_loader import build_feature_row
from src.test.conftest import live_data_available


def probability_to_risk(probability):
    probability = probability * 100

    if probability < 20:
        return "LOW"
    elif probability < 60:
        return "MEDIUM"
    else:
        return "HIGH"


def format_value(value, unit=""):
    if value is None:
        return "Unavailable"
    return f"{value:.1f}{unit}"


def evaluate_scenario(scenario_name, sample_data, weather_agent, radiation_agent):
    weather_result = weather_agent.predict(sample_data)
    radiation_result = radiation_agent.assess(sample_data)

    rf_probability = weather_result["frost_probability"] * 100
    rf_risk = probability_to_risk(weather_result["frost_probability"])
    radiation_risk = radiation_result["risk_level"]

    # Agreement is meaningful only when the RadiationFrostAgent has enough data.
    agreement = None if radiation_risk == "UNKNOWN" else rf_risk == radiation_risk

    print(f"\n{'-' * 60}\nScenario: {scenario_name}\n{'-' * 60}")
    print(f"\nRandom Forest -> Probability: {rf_probability:.1f}%, Risk: {rf_risk}")
    print(f"\nRadiation Frost -> Risk: {radiation_risk}")
    print("Minimum Air Temperature: " + format_value(radiation_result["air_temperature"], " °C"))
    print("Soil-Surface Temperature: " + format_value(radiation_result["soil_surface_temperature"], " °C"))
    print("Wind Speed: " + format_value(radiation_result["wind_speed"], " km/h"))
    print("Cloud Cover: " + format_value(radiation_result["cloud_cover"], "%"))

    if radiation_result["missing_inputs"]:
        print("Missing Radiation Inputs: " + ", ".join(radiation_result["missing_inputs"]))

    print(f"\nAgreement Between Methods: {'NOT COMPARABLE' if agreement is None else ('YES' if agreement else 'NO')}")

    return {
        "scenario": scenario_name,
        "rf_probability": rf_probability,
        "rf_risk": rf_risk,
        "radiation_risk": radiation_risk,
        "missing_inputs": radiation_result["missing_inputs"],
        "agreement": agreement,
    }


@pytest.mark.parametrize(
    "probability_pct,expected_risk",
    [
        (0.0, "LOW"),
        (19.9, "LOW"),
        (20.0, "MEDIUM"),
        (59.9, "MEDIUM"),
        (60.0, "HIGH"),
        (100.0, "HIGH"),
    ],
)
def test_probability_to_risk_boundaries(probability_pct, expected_risk):
    assert probability_to_risk(probability_pct / 100) == expected_risk


@pytest.fixture(scope="module")
def weather_agent():
    return WeatherAgent()


@pytest.fixture(scope="module")
def radiation_agent():
    return RadiationFrostAgent()


def test_historical_frost_event_is_high_rf_risk(weather_agent, radiation_agent):
    frost_sample = joblib.load("models/sample_row_frost.pkl")
    result = evaluate_scenario("Historical Frost Event", frost_sample, weather_agent, radiation_agent)

    assert result["rf_risk"] == "HIGH"
    # The bundled historical samples don't carry the radiation_* forecast fields,
    # so RadiationFrostAgent has nothing to compare the RF estimate against.
    assert result["radiation_risk"] == "UNKNOWN"
    assert result["agreement"] is None


def test_historical_recent_observation_is_low_rf_risk(weather_agent, radiation_agent):
    recent_sample = joblib.load("models/sample_row_recent.pkl")
    result = evaluate_scenario("Latest Historical Observation", recent_sample, weather_agent, radiation_agent)

    assert result["rf_risk"] == "LOW"
    assert result["radiation_risk"] == "UNKNOWN"
    assert result["agreement"] is None


@pytest.mark.skipif(not live_data_available(), reason="Open-Meteo API is not reachable")
def test_live_forecast_has_complete_radiation_inputs(weather_agent, radiation_agent):
    """Unlike the static historical samples, a live forecast row must carry every
    radiation_* field RadiationFrostAgent needs, making the two methods comparable."""
    live_sample = build_feature_row()
    result = evaluate_scenario("Live Open-Meteo Forecast", live_sample, weather_agent, radiation_agent)

    assert result["missing_inputs"] == []
    assert result["radiation_risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["agreement"] is not None


def main():
    weather_agent = WeatherAgent()
    radiation_agent = RadiationFrostAgent()

    print(f"\n{'=' * 60}\nFROST DETECTION COMPARISON\n{'=' * 60}")

    frost_sample = joblib.load("models/sample_row_frost.pkl")
    recent_sample = joblib.load("models/sample_row_recent.pkl")
    live_sample = build_feature_row()

    results = [
        evaluate_scenario("Historical Frost Event", frost_sample, weather_agent, radiation_agent),
        evaluate_scenario("Latest Historical Observation", recent_sample, weather_agent, radiation_agent),
        evaluate_scenario("Live Open-Meteo Forecast", live_sample, weather_agent, radiation_agent),
    ]

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for result in results:
        agreement_text = (
            "NOT COMPARABLE" if result["agreement"] is None else ("YES" if result["agreement"] else "NO")
        )
        print(f"\nScenario: {result['scenario']}")
        print(f"Random Forest Probability: {result['rf_probability']:.1f}%")
        print(f"Random Forest Risk: {result['rf_risk']}")
        print(f"Radiation Frost Risk: {result['radiation_risk']}")
        print(f"Agreement: {agreement_text}")


if __name__ == "__main__":
    main()
