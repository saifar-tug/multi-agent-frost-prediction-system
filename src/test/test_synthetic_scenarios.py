# src/test/test_synthetic_scenarios.py

from copy import deepcopy

import joblib
import pytest

from src.agents.weather_agent import WeatherAgent
from src.agents.soil_agent import SoilAgent
from src.agents.radiation_frost_agent import RadiationFrostAgent
from src.agents.planner_agent import PlannerAgent


def evaluate_scenario(title, sample_data, weather_agent, soil_agent, radiation_agent, planner_agent):
    weather_result = weather_agent.predict(sample_data)
    soil_result = soil_agent.assess(sample_data)
    radiation_result = radiation_agent.assess(sample_data)
    planner_result = planner_agent.plan(weather_result, soil_result, radiation_result)

    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print("\nSynthetic Conditions")
    print(f"Minimum Air Temperature: {radiation_result['air_temperature']:.1f} °C")

    soil_temperature = soil_result["soil_temperature"]
    if soil_temperature is not None:
        print(f"Soil-Surface Temperature: {soil_temperature:.1f} °C")

    print(f"Wind Speed: {radiation_result['wind_speed']:.1f} km/h")
    print(f"Cloud Cover: {radiation_result['cloud_cover']:.1f}%")

    print("\nAgent Assessments")
    print(f"WeatherAgent -> Frost Probability: {weather_result['frost_probability'] * 100:.1f}%, "
          f"Prediction: {weather_result['prediction_label']}")
    print(f"SoilAgent -> Soil Risk: {soil_result['soil_risk'].upper()}")
    print(f"RadiationFrostAgent -> Radiation Frost Risk: {radiation_result['risk_level']}")

    print("\nPlanner Decision")
    print(f"Evidence Status: {planner_result['evidence_status']}")
    print(f"Priority: {planner_result['priority']}")
    print(f"Recommended Action: {planner_result['recommended_action']}")
    print(f"Decision Basis: {planner_result['decision_basis']}")

    return {
        "weather": weather_result,
        "soil": soil_result,
        "radiation": radiation_result,
        "planner": planner_result,
    }


@pytest.fixture(scope="module")
def agents():
    return WeatherAgent(), SoilAgent(), RadiationFrostAgent(), PlannerAgent()


@pytest.fixture(scope="module")
def base_sample():
    return joblib.load("models/sample_row_frost.pkl")


def test_cold_but_cloudy_windy_yields_mixed_evidence(agents, base_sample):
    """Cold RF signal, but wind/cloud suppress the radiation-frost mechanism: conflicting evidence."""
    scenario = deepcopy(base_sample)
    scenario.update(
        temp_min=-1,
        radiation_temp_min=-1,
        radiation_soil_temp_min=-0.5,
        radiation_wind_speed=15,
        radiation_cloud_cover=80,
    )

    result = evaluate_scenario("SCENARIO 1 - COLD BUT CLOUDY/WINDY", scenario, *agents)

    assert result["soil"]["soil_risk"] == "high"
    assert result["radiation"]["risk_level"] == "LOW"
    assert result["planner"]["evidence_status"] == "MIXED"
    assert result["planner"]["priority"] == "MEDIUM"


def test_ideal_radiation_frost_conditions(agents, base_sample):
    """Cold, calm, clear, frozen soil surface: both physical specialists report HIGH risk."""
    scenario = deepcopy(base_sample)
    scenario.update(
        temp_min=0,
        radiation_temp_min=0,
        radiation_soil_temp_min=-1,
        radiation_wind_speed=2,
        radiation_cloud_cover=5,
    )

    result = evaluate_scenario("SCENARIO 2 - IDEAL RADIATION FROST", scenario, *agents)

    assert result["soil"]["soil_risk"] == "high"
    assert result["radiation"]["risk_level"] == "HIGH"
    # HIGH planner priority additionally requires RF frost_probability >= 0.50 when both
    # physical specialists agree (see PlannerAgent's high_support_count == 2 branch); a
    # moderate RF score alone keeps this at MEDIUM despite strong physical agreement.
    assert result["planner"]["priority"] == "MEDIUM"
    assert result["planner"]["evidence_status"] == "MIXED"


def test_borderline_conditions_trigger_monitoring(agents, base_sample):
    """Soil and radiation evidence are concerning but not severe: exercises the monitoring pathway."""
    scenario = deepcopy(base_sample)
    scenario.update(
        temp_min=1,
        radiation_temp_min=1,
        radiation_soil_temp_min=0.5,
        radiation_wind_speed=7,
        radiation_cloud_cover=35,
    )

    result = evaluate_scenario("SCENARIO 3 - BORDERLINE CASE", scenario, *agents)

    assert result["soil"]["soil_risk"] == "medium"
    assert result["radiation"]["risk_level"] == "MEDIUM"
    assert result["planner"]["priority"] == "MEDIUM"
    assert result["planner"]["recommended_action"] == (
        "Monitor conditions closely and prepare frost-protection measures."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
