# src/test/test_full_pipeline.py

from copy import deepcopy

import joblib
import pytest

from src.graph.frost_graph import frost_graph

SCENARIOS = [
    {
        "title": "Frost prediction",
        "question": "Will frost occur tomorrow?",
        "expected_agents": ["WeatherAgent", "SoilAgent"],
        "expected_intent": "frost_prediction",
    },
    {
        "title": "Protection decision",
        "question": "Should I activate frost protection tonight?",
        "expected_agents": ["WeatherAgent", "SoilAgent", "RadiationFrostAgent", "PlannerAgent"],
        "expected_intent": "protection_decision",
    },
    {
        "title": "Soil assessment",
        "question": "What is the soil condition?",
        "expected_agents": ["SoilAgent"],
        "expected_intent": "soil_assessment",
    },
    {
        "title": "Frost explanation",
        "question": "Explain radiation frost.",
        "expected_agents": [],
        "expected_intent": "frost_explanation",
    },
    {
        "title": "Out of scope",
        "question": "What is the oil price today?",
        "expected_agents": [],
        "expected_intent": "out_of_scope",
    },
    {
        "title": "Greeting",
        "question": "Hello",
        "expected_agents": [],
        "expected_intent": "greeting",
    },
]

AGENT_OUTPUT_KEYS = {
    "WeatherAgent": "weather_output",
    "SoilAgent": "soil_output",
    "RadiationFrostAgent": "radiation_output",
    "PlannerAgent": "planner_output",
}


@pytest.fixture(scope="module")
def sample_data():
    """A fixed frost-risk scenario, not live Open-Meteo data, so results are reproducible."""
    sample = deepcopy(joblib.load("models/sample_row_frost.pkl"))

    sample["data_source"] = "Controlled Full-Pipeline Test"
    sample["location"] = "Graz, Austria"

    sample["temp_min"] = 0.0
    sample["radiation_temp_min"] = 0.0
    sample["radiation_soil_temp_min"] = -1.0
    sample["radiation_wind_speed"] = 2.0
    sample["radiation_cloud_cover"] = 5.0

    return sample


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["title"] for s in SCENARIOS])
def test_full_pipeline_scenario(sample_data, scenario):
    result = frost_graph.invoke({"question": scenario["question"], "sample_data": deepcopy(sample_data)})

    query_understanding = result.get("query_understanding", {})
    required_agents = result.get("required_agents", [])
    final_response = result.get("final_response")

    assert required_agents == scenario["expected_agents"]
    assert query_understanding.get("intent") == scenario["expected_intent"]
    assert final_response is not None and final_response.strip()

    planner_output = result.get("planner_output")
    if planner_output is not None:
        recommended_action = planner_output.get("recommended_action")
        assert recommended_action is not None
        assert recommended_action.lower() in final_response.lower()

    # Only the agents this scenario actually needs should have run.
    for agent_name, output_key in AGENT_OUTPUT_KEYS.items():
        output = result.get(output_key)
        if agent_name in scenario["expected_agents"]:
            assert output is not None, f"{agent_name} should have produced output"
        else:
            assert output is None, f"{agent_name} was unnecessarily executed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
