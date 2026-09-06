import pytest

from src.agents.query_understanding_agent import QueryUnderstandingAgent
from src.config import LLM_MODEL_NAME
from src.test.conftest import ollama_available

agent = QueryUnderstandingAgent(model_name=LLM_MODEL_NAME)
OLLAMA_AVAILABLE = ollama_available()

# Canonical phrasings the deterministic classifier resolves without the LLM fallback.
DETERMINISTIC_CASES = [
    ("Will frost occur tomorrow?", "frost", "frost_prediction"),
    ("What is the soil condition?", "frost", "soil_assessment"),
    ("Should I activate frost protection tonight?", "frost", "protection_decision"),
    ("Explain radiation frost.", "frost", "frost_explanation"),
    ("Will there be frost in Graz tomorrow?", "frost", "frost_prediction"),
    ("What is the oil price today?", "out_of_scope", "out_of_scope"),
    ("What is the temperature of Vienna?", "out_of_scope", "out_of_scope"),
    ("Will it rain tomorrow?", "out_of_scope", "out_of_scope"),
    ("What is the weather in Graz?", "out_of_scope", "out_of_scope"),
    ("Hello", "general_conversation", "greeting"),
]

# Natural paraphrases that only the LLM fallback can classify; skipped unless
# a local Ollama server is reachable.
LLM_DEPENDENT_CASES = [
    ("Do I need to worry about my plants freezing overnight?", "frost"),
    ("Could tonight's cold damage the vineyard?", "frost"),
    ("Do these conditions look dangerous for my crops?", "frost"),
    ("Is the ground likely to get cold enough to harm my plants?", "frost"),
    ("Do I need to cover my plants tonight?", "frost"),
    ("Is it safe to leave sensitive plants outside overnight?", "frost"),
]


@pytest.mark.parametrize("question,expected_domain,expected_intent", DETERMINISTIC_CASES)
def test_deterministic_classification(question, expected_domain, expected_intent):
    result = agent.understand(question)
    assert result["domain"] == expected_domain
    assert result["intent"] == expected_intent


@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama is not running locally")
@pytest.mark.parametrize("question,expected_domain", LLM_DEPENDENT_CASES)
def test_llm_assisted_classification(question, expected_domain):
    result = agent.understand(question)
    assert result["domain"] == expected_domain


def test_time_reference_extraction():
    assert agent.understand("Will frost occur tomorrow?")["entities"]["time_reference"] == "tomorrow"
    assert agent.understand("Should I activate frost protection tonight?")["entities"]["time_reference"] == "tonight"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
