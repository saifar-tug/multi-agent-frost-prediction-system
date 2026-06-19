# src/graph/frost_graph.py

from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph import END

from src.agents.weather_agent import WeatherAgent
from src.agents.soil_agent import SoilAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.llm_orchestrator import LLMOrchestrator


class FrostState(TypedDict):

    question: str

    sample_data: dict

    required_agents: list[str]

    weather_output: dict | None

    soil_output: dict | None

    planner_output: dict | None

    final_response: str | None


weather_agent = WeatherAgent()

soil_agent = SoilAgent()

planner_agent = PlannerAgent()

llm_orchestrator = LLMOrchestrator(
    model_name="llama3:latest"
)


def agent_selection_node(state):

    question = (
        state["question"]
        .lower()
        .strip()
    )

    if any(
        keyword in question
        for keyword in [
            "action",
            "protect",
            "protection",
            "heater",
            "fan",
            "then",
            "recommend",
            "recommendation",
            "should i",
            "should we",
            "what should i do",
            "what should we do",
            "plan",
            "decision",
            "advice",
            "advise",
            "suggestion",
            "what do you recommend",
            "recommended action"
        ]
    ):

        required_agents = [
            "WeatherAgent",
            "SoilAgent",
            "PlannerAgent"
        ]

    elif any(
        keyword in question
        for keyword in [
            "soil",
            "ground"
        ]
    ):

        required_agents = [
            "SoilAgent"
        ]

    elif any(
        keyword in question
        for keyword in [
            "frost",
            "weather",
            "forecast",
            "temperature",
            "tomorrow",
            "tonight"
        ]
    ):

        required_agents = [
            "WeatherAgent",
            "SoilAgent"
        ]

    elif any(
        keyword in question
        for keyword in [
            "hi",
            "hello",
            "hey",
            "help",
            "who are you",
            "what can you do",
            "introduce yourself",
            "clear"
        ]
    ):

        required_agents = [
            "GreetingAgent"
        ]

    elif any(
        keyword in question
        for keyword in [
            "okay",
            "ok",
            "thanks",
            "thank you",
            "yes",
            "no",
            "cool",
            "great"
        ]
    ):

        required_agents = [
            "ConversationAgent"
        ]

    else:

        required_agents = [
            "WeatherAgent",
            "SoilAgent"
        ]

    return {
        "required_agents": required_agents
    }


def weather_node(state):

    if (
        "WeatherAgent"
        not in state["required_agents"]
    ):

        return {}

    weather_output = (
        weather_agent.predict(
            state["sample_data"]
        )
    )

    return {
        "weather_output": weather_output
    }


def soil_node(state):

    if (
        "SoilAgent"
        not in state["required_agents"]
    ):

        return {}

    soil_output = (
        soil_agent.assess(
            state["sample_data"]
        )
    )

    return {
        "soil_output": soil_output
    }


def planner_node(state):

    if (
        "PlannerAgent"
        not in state["required_agents"]
    ):

        return {}

    weather_output = state.get(
        "weather_output"
    )

    soil_output = state.get(
        "soil_output"
    )

    if weather_output is None:

        weather_output = (
            weather_agent.predict(
                state["sample_data"]
            )
        )

    if soil_output is None:

        soil_output = (
            soil_agent.assess(
                state["sample_data"]
            )
        )

    planner_output = (
        planner_agent.plan(
            weather_output,
            soil_output
        )
    )

    return {
        "weather_output": weather_output,
        "soil_output": soil_output,
        "planner_output": planner_output
    }


def response_node(state):

    weather_output = state.get(
        "weather_output"
    )

    soil_output = state.get(
        "soil_output"
    )

    planner_output = state.get(
        "planner_output"
    )

    sample_data = state[
        "sample_data"
    ]

    if state["required_agents"] == ["GreetingAgent"]:

        final_response = """
    Hello. I am a Frost Risk Decision Support Assistant.

    I can help with:

    • Frost prediction
    • Soil condition assessment
    • Frost protection recommendations
    • Real-time weather-based frost risk analysis

    Example questions:

    • Will frost occur tomorrow?
    • What is the soil temperature?
    • Should I activate frost protection tonight?
    """

        return {
            "final_response": final_response
        }

    if state["required_agents"] == ["ConversationAgent"]:

        final_response = """
    Understood.

    You can ask me about frost risk, soil temperature, or frost protection decisions.

    Example:
    • Will frost occur tomorrow?
    • Should I activate frost protection tonight?
    """

        return {
            "final_response": final_response
        }

    if planner_output is not None:

        final_response = (
            llm_orchestrator.explain(
                weather_output,
                soil_output,
                planner_output
            )
        )

    elif weather_output is not None and soil_output is not None:

        frost_probability = (
            weather_output[
                "frost_probability"
            ] * 100
        )

        frost_prediction = (
            "Yes"
            if weather_output[
                "frost_prediction"
            ] == 1
            else "No"
        )

        if frost_probability < 20:

            risk_level = "LOW"

        elif frost_probability < 60:

            risk_level = "MEDIUM"

        else:

            risk_level = "HIGH"

        recommendation = (
            "Activate frost protection immediately."
            if weather_output[
                "frost_prediction"
            ] == 1
            else
            "Continue normal operation. No frost protection measures are currently required."
        )

        final_response = f"""
CURRENT FROST ASSESSMENT

Source:
{sample_data.get('data_source', 'GeoSphere Austria')}

Location:
{sample_data.get('location', 'Graz, Austria')}

Forecast Generated:
{sample_data.get('forecast_generated', 'N/A')}

Prediction Date:
{sample_data.get('prediction_date', 'N/A')}

Frost Probability:
{frost_probability:.1f}%

Risk Level:
{risk_level}

Predicted Frost Event:
{frost_prediction}

Soil Temperature:
{soil_output['soil_temperature']:.1f} °C

Soil Risk:
{soil_output['soil_risk'].title()}

Recommendation:
{recommendation}
"""

    elif soil_output is not None:

        final_response = (
            "SOIL ASSESSMENT\n\n"
            f"Soil Temperature: "
            f"{soil_output['soil_temperature']:.1f} °C\n\n"
            f"Soil Risk: "
            f"{soil_output['soil_risk'].title()}\n\n"
            f"Explanation: "
            f"{soil_output['explanation']}"
        )

    elif weather_output is not None:

        frost_probability = (
            weather_output[
                "frost_probability"
            ] * 100
        )

        frost_prediction = (
            "Yes"
            if weather_output[
                "frost_prediction"
            ] == 1
            else "No"
        )

        final_response = f"""
WEATHER ASSESSMENT

Frost Probability:
{frost_probability:.1f}%

Predicted Frost Event:
{frost_prediction}
"""

    else:

        final_response = (
            "No relevant agent output was produced."
        )

    return {
        "final_response": final_response
    }


graph = StateGraph(
    FrostState
)

graph.add_node(
    "agent_selection",
    agent_selection_node
)

graph.add_node(
    "weather",
    weather_node
)

graph.add_node(
    "soil",
    soil_node
)

graph.add_node(
    "planner",
    planner_node
)

graph.add_node(
    "response",
    response_node
)

graph.set_entry_point(
    "agent_selection"
)

graph.add_edge(
    "agent_selection",
    "weather"
)

graph.add_edge(
    "weather",
    "soil"
)

graph.add_edge(
    "soil",
    "planner"
)

graph.add_edge(
    "planner",
    "response"
)

graph.add_edge(
    "response",
    END
)

frost_graph = graph.compile()