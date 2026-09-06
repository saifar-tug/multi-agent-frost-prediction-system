# src/test/test_planner_policy.py

import pytest

from src.agents.planner_agent import PlannerAgent

TEST_CASES = [
    {
        "name": "All evidence low",
        "frost_probability": 0.10,
        "soil_risk": "LOW",
        "radiation_risk": "LOW",
        "expected_priority": "LOW",
        "expected_action": "No immediate action required.",
        "expected_status": "AGREEMENT",
    },
    {
        "name": "Moderate RF evidence only",
        "frost_probability": 0.55,
        "soil_risk": "LOW",
        "radiation_risk": "LOW",
        "expected_priority": "MEDIUM",
        "expected_action": "Monitor conditions closely and prepare frost-protection measures.",
        "expected_status": "MIXED",
    },
    {
        "name": "Moderate RF plus two high physical warnings",
        "frost_probability": 0.55,
        "soil_risk": "HIGH",
        "radiation_risk": "HIGH",
        "expected_priority": "HIGH",
        "expected_action": "Activate frost protection.",
        "expected_status": "AGREEMENT",
    },
    {
        "name": "Strong RF plus high soil warning",
        "frost_probability": 0.75,
        "soil_risk": "HIGH",
        "radiation_risk": "LOW",
        "expected_priority": "HIGH",
        "expected_action": "Activate frost protection.",
        "expected_status": "MIXED",
    },
    {
        "name": "Strong RF plus high radiation warning",
        "frost_probability": 0.75,
        "soil_risk": "LOW",
        "radiation_risk": "HIGH",
        "expected_priority": "HIGH",
        "expected_action": "Activate frost protection.",
        "expected_status": "MIXED",
    },
    {
        "name": "Low RF with two medium physical warnings",
        "frost_probability": 0.40,
        "soil_risk": "MEDIUM",
        "radiation_risk": "MEDIUM",
        "expected_priority": "MEDIUM",
        "expected_action": "Monitor conditions closely and prepare frost-protection measures.",
        "expected_status": "MIXED",
    },
    {
        "name": "Low RF with high radiation warning",
        "frost_probability": 0.10,
        "soil_risk": "LOW",
        "radiation_risk": "HIGH",
        "expected_priority": "MEDIUM",
        "expected_action": "Monitor conditions closely and prepare frost-protection measures.",
        "expected_status": "MIXED",
    },
    {
        "name": "Low RF with high soil warning",
        "frost_probability": 0.10,
        "soil_risk": "HIGH",
        "radiation_risk": "LOW",
        "expected_priority": "MEDIUM",
        "expected_action": "Monitor conditions closely and prepare frost-protection measures.",
        "expected_status": "MIXED",
    },
]


def build_weather_output(frost_probability):
    return {
        "agent": "WeatherAgent",
        "frost_probability": frost_probability,
        "frost_prediction": int(frost_probability >= 0.5),
    }


def build_soil_output(soil_risk):
    return {"agent": "SoilAgent", "soil_risk": soil_risk}


def build_radiation_output(radiation_risk):
    return {"agent": "RadiationFrostAgent", "risk_level": radiation_risk}


@pytest.fixture(scope="module")
def planner():
    return PlannerAgent()


@pytest.mark.parametrize("test_case", TEST_CASES, ids=[case["name"] for case in TEST_CASES])
def test_planner_policy(planner, test_case):
    result = planner.plan(
        build_weather_output(test_case["frost_probability"]),
        build_soil_output(test_case["soil_risk"]),
        build_radiation_output(test_case["radiation_risk"]),
    )

    assert result["priority"] == test_case["expected_priority"]
    assert result["recommended_action"] == test_case["expected_action"]
    assert result["evidence_status"] == test_case["expected_status"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
