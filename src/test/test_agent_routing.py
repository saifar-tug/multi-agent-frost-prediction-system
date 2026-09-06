# src/test/test_agent_routing.py

import pytest

from src.agents.llm_router_agent import LLMRouterAgent

ROUTING_CASES = [
    (
        "Frost prediction",
        {"domain": "frost", "intent": "frost_prediction"},
        ["WeatherAgent", "SoilAgent"],
    ),
    (
        "Soil assessment",
        {"domain": "frost", "intent": "soil_assessment"},
        ["SoilAgent"],
    ),
    (
        "Radiation frost assessment",
        {"domain": "frost", "intent": "radiation_frost_assessment"},
        ["RadiationFrostAgent"],
    ),
    (
        "Protection decision",
        {"domain": "frost", "intent": "protection_decision"},
        ["WeatherAgent", "SoilAgent", "RadiationFrostAgent", "PlannerAgent"],
    ),
    (
        "Frost explanation",
        {"domain": "frost", "intent": "frost_explanation"},
        [],
    ),
    (
        "Greeting",
        {"domain": "general_conversation", "intent": "greeting"},
        [],
    ),
    (
        "Out of scope",
        {"domain": "out_of_scope", "intent": "out_of_scope"},
        [],
    ),
    (
        "Unknown request",
        {"domain": "unknown", "intent": "unknown"},
        [],
    ),
]


@pytest.fixture(scope="module")
def router():
    return LLMRouterAgent()


@pytest.mark.parametrize(
    "name,query_understanding,expected_agents",
    ROUTING_CASES,
    ids=[case[0] for case in ROUTING_CASES],
)
def test_agent_selection(router, name, query_understanding, expected_agents):
    assert router.select_agents(query_understanding) == expected_agents


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
