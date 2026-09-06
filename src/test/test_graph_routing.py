# src/test/test_graph_routing.py

import pytest

from src.data_pipeline.live_data_loader import build_feature_row
from src.graph.frost_graph import frost_graph
from src.test.conftest import live_data_available

ROUTING_CASES = [
    ("Greeting", "Hello", ["query_understanding", "agent_selection", "response"]),
    (
        "Out-of-scope request",
        "What is the oil price today?",
        ["query_understanding", "agent_selection", "response"],
    ),
    (
        "Frost explanation",
        "Explain radiation frost.",
        ["query_understanding", "agent_selection", "response"],
    ),
    (
        "Frost prediction",
        "Will frost occur tomorrow?",
        ["query_understanding", "agent_selection", "weather", "soil", "response"],
    ),
    (
        "Soil assessment",
        "What is the soil condition?",
        ["query_understanding", "agent_selection", "soil", "response"],
    ),
    (
        "Protection decision",
        "Should I activate frost protection tonight?",
        ["query_understanding", "agent_selection", "weather", "soil", "radiation", "planner", "response"],
    ),
]


def run_routing_test(question, sample_data):
    initial_state = {"question": question, "sample_data": sample_data}

    executed_nodes = []
    for event in frost_graph.stream(initial_state, stream_mode="updates"):
        executed_nodes.extend(event.keys())

    return executed_nodes


@pytest.fixture(scope="module")
def sample_data():
    if not live_data_available():
        pytest.skip("Open-Meteo API is not reachable")
    return build_feature_row()


@pytest.mark.parametrize(
    "name,question,expected_nodes",
    ROUTING_CASES,
    ids=[case[0] for case in ROUTING_CASES],
)
def test_graph_executes_expected_nodes(sample_data, name, question, expected_nodes):
    executed_nodes = run_routing_test(question, sample_data)
    assert executed_nodes == expected_nodes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
