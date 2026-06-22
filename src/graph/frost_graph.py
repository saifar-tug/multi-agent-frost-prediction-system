# src/graph/frost_graph.py

from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.weather_agent import WeatherAgent
from agents.soil_agent import SoilAgent
from agents.radiation_frost_agent import RadiationFrostAgent
from agents.planner_agent import PlannerAgent
from agents.llm_orchestrator import LLMOrchestrator
from agents.llm_router_agent import LLMRouterAgent


class FrostState(TypedDict, total=False):

    question: str
    sample_data: dict

    required_agents: list[str]

    weather_output: dict | None
    soil_output: dict | None
    radiation_output: dict | None
    planner_output: dict | None

    final_response: str | None


weather_agent = WeatherAgent()
soil_agent = SoilAgent()
radiation_agent = RadiationFrostAgent()
planner_agent = PlannerAgent()

router_agent = LLMRouterAgent(
    model_name="llama3:latest"
)

llm_orchestrator = LLMOrchestrator(
    model_name="llama3:latest"
)


def agent_selection_node(state):

    required_agents = router_agent.select_agents(
        state["question"]
    )

    return {
        "required_agents": required_agents
    }


def weather_node(state):

    if "WeatherAgent" not in state["required_agents"]:

        return {}

    weather_output = weather_agent.predict(
        state["sample_data"]
    )

    return {
        "weather_output": weather_output
    }


def soil_node(state):

    if "SoilAgent" not in state["required_agents"]:

        return {}

    soil_output = soil_agent.assess(
        state["sample_data"]
    )

    return {
        "soil_output": soil_output
    }


def radiation_node(state):

    if "RadiationFrostAgent" not in state["required_agents"]:

        return {}

    radiation_output = radiation_agent.assess(
        state["sample_data"]
    )

    return {
        "radiation_output": radiation_output
    }


def planner_node(state):

    if "PlannerAgent" not in state["required_agents"]:

        return {}

    weather_output = state.get("weather_output")
    soil_output = state.get("soil_output")
    radiation_output = state.get("radiation_output")

    if weather_output is None:

        weather_output = weather_agent.predict(
            state["sample_data"]
        )

    if soil_output is None:

        soil_output = soil_agent.assess(
            state["sample_data"]
        )

    if radiation_output is None:

        radiation_output = radiation_agent.assess(
            state["sample_data"]
        )

    try:

        planner_output = planner_agent.plan(
            weather_output,
            soil_output,
            radiation_output
        )

    except TypeError:

        planner_output = planner_agent.plan(
            weather_output,
            soil_output
        )

        planner_output["radiation_output"] = radiation_output

    return {
        "weather_output": weather_output,
        "soil_output": soil_output,
        "radiation_output": radiation_output,
        "planner_output": planner_output
    }


def _get_risk_level(frost_probability):

    if frost_probability < 20:

        return "LOW"

    if frost_probability < 60:

        return "MEDIUM"

    return "HIGH"


def _get_frost_prediction_label(weather_output):

    return (
        "Yes"
        if weather_output["frost_prediction"] == 1
        else "No"
    )


def response_node(state):

    required_agents = state.get(
        "required_agents",
        []
    )

    weather_output = state.get("weather_output")
    soil_output = state.get("soil_output")
    radiation_output = state.get("radiation_output")
    planner_output = state.get("planner_output")

    sample_data = state["sample_data"]

    if required_agents == ["GreetingAgent"]:

        final_response = """
Hello. I am a Frost Risk Decision Support Assistant.

I can help with:
- Frost prediction using a trained Random Forest model
- Soil condition assessment
- Radiation frost assessment
- Frost protection recommendations
- Real-time weather-based frost risk analysis

You can ask:
- Will frost occur tomorrow?
- What is the soil temperature?
- Should I activate frost protection tonight?
"""

        return {
            "final_response": final_response
        }

    if required_agents == ["ConversationAgent"]:

        final_response = """
Understood.

You can ask me about:
- frost risk
- soil temperature
- radiation frost conditions
- frost protection decisions
"""

        return {
            "final_response": final_response
        }

    if planner_output is not None:

        final_response = llm_orchestrator.explain(
            weather_output,
            soil_output,
            planner_output
        )

        return {
            "final_response": final_response
        }

    if weather_output is not None and soil_output is not None:

        frost_probability = (
            weather_output["frost_probability"] * 100
        )

        risk_level = _get_risk_level(
            frost_probability
        )

        frost_prediction = _get_frost_prediction_label(
            weather_output
        )

        final_response = f"""
FROST RISK ASSESSMENT

Data Source:
GeoSphere Austria (Graz Universität Station)

Analysis Context:
Historical Frost Event from the GeoSphere weather dataset

Location:
Graz, Austria

Model Used:
Random Forest frost prediction model

Frost Probability:
{frost_probability:.1f}%

Risk Level:
{risk_level}

Predicted Frost Event:
{frost_prediction}

Soil Temperature:
{soil_output["soil_temperature"]:.1f} °C

Soil Risk:
{soil_output["soil_risk"].title()}

Assessment Summary:
The trained Random Forest model predicts a {frost_probability:.1f}% probability of frost for this historical weather record. The soil assessment also indicates {soil_output["soil_risk"].lower()} ground-level risk. Together, these outputs indicate a {risk_level.lower()} frost-risk situation.

Decision Note:
No operational recommendation was generated because this question requested a frost-risk assessment, not an action plan. PlannerAgent was therefore not activated.
"""

        return {
            "final_response": final_response
        }

    if radiation_output is not None:

        final_response = f"""
RADIATION FROST ASSESSMENT

Data Source:
GeoSphere Austria (Graz Universität Station)

Analysis Context:
Historical Frost Event from the GeoSphere weather dataset

Location:
Graz, Austria

Temperature:
{radiation_output.get("temperature", sample_data.get("temp_min", "N/A"))} °C

Wind Speed:
{radiation_output.get("wind_speed", sample_data.get("max_wind_gust", "N/A"))} km/h

Cloud Cover:
{radiation_output.get("cloud_cover", sample_data.get("cloud_morning", "N/A"))} %

Radiation Frost Risk:
{radiation_output.get("radiation_frost_risk", radiation_output.get("risk_level", "N/A"))}

Assessment Summary:
{radiation_output.get("explanation", "Radiation frost conditions were assessed using temperature, wind, and cloud-cover information.")}
"""

        return {
            "final_response": final_response
        }

    if soil_output is not None:

        final_response = f"""
SOIL CONDITION ASSESSMENT

Data Source:
GeoSphere Austria (Graz Universität Station)

Analysis Context:
Historical Frost Event from the GeoSphere weather dataset

Location:
Graz, Austria

Soil Temperature:
{soil_output["soil_temperature"]:.1f} °C

Soil Risk:
{soil_output["soil_risk"].title()}

Assessment Summary:
{soil_output["explanation"]}
"""

        return {
            "final_response": final_response
        }

    if weather_output is not None:

        frost_probability = (
            weather_output["frost_probability"] * 100
        )

        risk_level = _get_risk_level(
            frost_probability
        )

        frost_prediction = _get_frost_prediction_label(
            weather_output
        )

        final_response = f"""
WEATHER-BASED FROST ASSESSMENT

Data Source:
GeoSphere Austria (Graz Universität Station)

Analysis Context:
Historical Frost Event from the GeoSphere weather dataset

Location:
Graz, Austria

Model Used:
Random Forest frost prediction model

Frost Probability:
{frost_probability:.1f}%

Risk Level:
{risk_level}

Predicted Frost Event:
{frost_prediction}

Assessment Summary:
The trained Random Forest model predicts a {frost_probability:.1f}% probability of frost for this historical weather record.

Decision Note:
No soil, radiation, or planning assessment was generated because only WeatherAgent was activated.
"""

        return {
            "final_response": final_response
        }

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
    "radiation",
    radiation_node
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
    "radiation"
)

graph.add_edge(
    "radiation",
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