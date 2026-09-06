# src/graph/frost_graph.py

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.weather_agent import WeatherAgent
from src.agents.soil_agent import SoilAgent
from src.agents.radiation_frost_agent import RadiationFrostAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.llm_orchestrator import LLMOrchestrator
from src.agents.llm_router_agent import LLMRouterAgent
from src.agents.query_understanding_agent import (
    QueryUnderstandingAgent,
)
from src.config import LLM_MODEL_NAME


class FrostState(TypedDict, total=False):

    question: str

    sample_data: dict[str, Any]

    query_understanding: dict[str, Any]

    required_agents: list[str]

    weather_output: dict[str, Any] | None

    soil_output: dict[str, Any] | None

    radiation_output: dict[str, Any] | None

    planner_output: dict[str, Any] | None

    final_response: str | None


query_understanding_agent = QueryUnderstandingAgent(
    model_name=LLM_MODEL_NAME,
)

router_agent = LLMRouterAgent()

weather_agent = WeatherAgent()

soil_agent = SoilAgent()

radiation_agent = RadiationFrostAgent()

planner_agent = PlannerAgent()

llm_orchestrator = LLMOrchestrator(
    model_name=LLM_MODEL_NAME
)


def query_understanding_node(
    state: FrostState,
) -> dict[str, Any]:

    query_understanding = (
        query_understanding_agent.understand(
            state["question"]
        )
    )

    return {
        "query_understanding":
            query_understanding
    }


def agent_selection_node(
    state: FrostState,
) -> dict[str, Any]:

    query_understanding = state[
        "query_understanding"
    ]

    required_agents = (
        router_agent.select_agents(
            query_understanding
        )
    )

    return {
        "required_agents":
            required_agents
    }


def weather_node(
    state: FrostState,
) -> dict[str, Any]:

    weather_output = (
        weather_agent.predict(
            state["sample_data"]
        )
    )

    return {
        "weather_output":
            weather_output
    }


def soil_node(
    state: FrostState,
) -> dict[str, Any]:

    soil_output = (
        soil_agent.assess(
            state["sample_data"]
        )
    )

    return {
        "soil_output":
            soil_output
    }


def radiation_node(
    state: FrostState,
) -> dict[str, Any]:

    radiation_output = (
        radiation_agent.assess(
            state["sample_data"]
        )
    )

    return {
        "radiation_output":
            radiation_output
    }


def planner_node(
    state: FrostState,
) -> dict[str, Any]:

    weather_output = state.get(
        "weather_output"
    )

    soil_output = state.get(
        "soil_output"
    )

    radiation_output = state.get(
        "radiation_output"
    )

    if weather_output is None:
        raise RuntimeError(
            "PlannerAgent requires WeatherAgent output, "
            "but weather_output is missing."
        )

    if soil_output is None:
        raise RuntimeError(
            "PlannerAgent requires SoilAgent output, "
            "but soil_output is missing."
        )

    if radiation_output is None:
        raise RuntimeError(
            "PlannerAgent requires RadiationFrostAgent output, "
            "but radiation_output is missing."
        )

    planner_output = planner_agent.plan(
        weather_output,
        soil_output,
        radiation_output,
    )

    return {
        "planner_output": planner_output,
    }


def _get_risk_level(
    frost_probability: float,
) -> str:

    if frost_probability < 20:

        return "LOW"

    if frost_probability < 60:

        return "MEDIUM"

    return "HIGH"


def _get_frost_prediction_label(
    weather_output: dict[str, Any],
) -> str:

    if (
        weather_output[
            "frost_prediction"
        ]
        == 1
    ):

        return "Yes"

    return "No"


def _get_metadata(
    sample_data: dict[str, Any],
) -> dict[str, Any]:

    data_source = sample_data.get(
        "data_source",
        "GeoSphere Austria",
    )

    location = sample_data.get(
        "location",
        "Graz, Austria",
    )

    forecast_generated = (
        sample_data.get(
            "forecast_generated"
        )
    )

    prediction_date = (
        sample_data.get(
            "prediction_date"
        )
    )

    forecast_window_start = (
        sample_data.get(
            "forecast_window_start"
        )
    )

    forecast_window_end = (
        sample_data.get(
            "forecast_window_end"
        )
    )

    if (
        "open-meteo"
        in str(data_source).lower()
    ):

        analysis_context = (
            "Operational Live Weather Forecast"
        )

    else:

        analysis_context = (
            "Historical Weather Observation"
        )

    return {
        "data_source":
            data_source,

        "location":
            location,

        "forecast_generated":
            forecast_generated,

        "prediction_date":
            prediction_date,

        "forecast_window_start":
            forecast_window_start,

        "forecast_window_end":
            forecast_window_end,

        "analysis_context":
            analysis_context,
    }


def _build_metadata_text(
    sample_data: dict[str, Any],
) -> str:

    metadata = _get_metadata(
        sample_data
    )

    lines = [

        "Data Source:",
        str(
            metadata[
                "data_source"
            ]
        ),

        "",
        "Analysis Context:",
        str(
            metadata[
                "analysis_context"
            ]
        ),

        "",
        "Location:",
        str(
            metadata[
                "location"
            ]
        ),
    ]

    if (
        metadata[
            "forecast_generated"
        ]
        is not None
    ):

        lines.extend([
            "",
            "Forecast Generated:",
            str(
                metadata[
                    "forecast_generated"
                ]
            ),
        ])

    if (
        metadata[
            "prediction_date"
        ]
        is not None
    ):

        lines.extend([
            "",
            "Prediction Date:",
            str(
                metadata[
                    "prediction_date"
                ]
            ),
        ])

    if (
        metadata[
            "forecast_window_start"
        ]
        is not None
        and
        metadata[
            "forecast_window_end"
        ]
        is not None
    ):

        lines.extend([
            "",
            "Forecast Window:",
            (
                f"{metadata['forecast_window_start']} "
                f"to "
                f"{metadata['forecast_window_end']}"
            ),
        ])

    return "\n".join(
        lines
    )


def response_node(
    state: FrostState,
) -> dict[str, Any]:

    query_understanding = (
        state.get(
            "query_understanding",
            {},
        )
    )

    domain = (
        query_understanding.get(
            "domain",
            "unknown",
        )
    )

    intent = (
        query_understanding.get(
            "intent",
            "unknown",
        )
    )

    confidence = (
        query_understanding.get(
            "confidence",
            0.0,
        )
    )

    weather_output = state.get(
        "weather_output"
    )

    soil_output = state.get(
        "soil_output"
    )

    radiation_output = state.get(
        "radiation_output"
    )

    planner_output = state.get(
        "planner_output"
    )

    sample_data = state[
        "sample_data"
    ]

    metadata_text = (
        _build_metadata_text(
            sample_data
        )
    )

    # Out-of-scope request

    if domain == "out_of_scope":

        topic = (
            query_understanding
            .get(
                "entities",
                {},
            )
            .get(
                "topic"
            )
        )

        topic_text = (
            f" ({topic})"
            if topic
            else ""
        )

        final_response = f"""
OUT-OF-SCOPE REQUEST

This system is specialized in frost prediction,
soil frost-risk assessment, radiation-frost analysis,
and frost-protection decision support.

Your request{topic_text} is outside the supported domain.

No frost-analysis agents or prediction models were activated.
"""

        return {
            "final_response":
                final_response
        }

    # Unknown / ambiguous request

    if (
        domain == "unknown"
        or intent == "unknown"
    ):

        final_response = f"""
REQUEST NOT UNDERSTOOD

The system could not confidently classify this request.

Query Understanding Confidence:
{confidence * 100:.1f}%

Please rephrase the question in the context of frost prediction,
soil conditions, radiation frost, or frost-protection decisions.

No frost-analysis agents were activated.
"""

        return {
            "final_response":
                final_response
        }

    # Greeting

    if intent == "greeting":

        final_response = """
Hello. I am a Multi-Agent Frost Prediction and Decision Support System.

I can assist with:

- frost prediction
- soil frost-risk assessment
- radiation-frost conditions
- frost-protection decisions
- operational weather-based frost analysis
"""

        return {
            "final_response":
                final_response
        }

    # Acknowledgement

    if intent == "acknowledgement":

        final_response = """
Understood.

You can continue by asking about frost probability,
soil conditions, radiation frost, or frost-protection decisions.
"""

        return {
            "final_response":
                final_response
        }

    # Conceptual Frost Explanation

    if intent == "frost_explanation":

        final_response = """
FROST EXPLANATION

Frost develops when surfaces cool sufficiently for water vapour
to deposit or freeze near the ground.

Operational frost risk depends on several interacting factors,
including air temperature, ground-surface temperature, humidity,
cloud cover, wind conditions, and nighttime radiative cooling.

This request was informational, so no prediction model or
operational decision agent was activated.
"""

        return {
            "final_response":
                final_response
        }

    # Planner / operational decision

    if planner_output is not None:

        final_response = (
            llm_orchestrator.explain(
                weather_output,
                soil_output,
                radiation_output,
                planner_output
            )
        )

        return {
            "final_response":
                final_response
        }

    # Weather + Soil frost assessment

    if (
        weather_output is not None
        and soil_output is not None
    ):

        frost_probability = (
            weather_output[
                "frost_probability"
            ]
            * 100
        )

        risk_level = (
            _get_risk_level(
                frost_probability
            )
        )

        frost_prediction = (
            _get_frost_prediction_label(
                weather_output
            )
        )

        final_response = f"""
FROST RISK ASSESSMENT

{metadata_text}

Model Used:
Random Forest Frost Prediction Model

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
The trained Random Forest model estimates a
{frost_probability:.1f}% probability of frost for the evaluated
forecast conditions.

The SoilAgent independently classifies the ground-level risk as
{soil_output["soil_risk"].lower()}.

Together, these assessments indicate a
{risk_level.lower()} frost-risk situation.

Decision Note:
This request asked for a frost assessment rather than an
operational action. PlannerAgent was therefore not activated.
"""

        return {
            "final_response":
                final_response
        }

    # Radiation Frost Assessment

    if radiation_output is not None:

        final_response = f"""
RADIATION FROST ASSESSMENT

{metadata_text}

Air Temperature:
{radiation_output.get(
    "temperature",
    sample_data.get(
        "radiation_temp_min",
        sample_data.get(
            "temp_min",
            "N/A",
        ),
    ),
)} °C

Wind Speed:
{radiation_output.get(
    "wind_speed",
    sample_data.get(
        "radiation_wind_speed",
        "N/A",
    ),
)} km/h

Cloud Cover:
{radiation_output.get(
    "cloud_cover",
    sample_data.get(
        "radiation_cloud_cover",
        "N/A",
    ),
)} %

Radiation Frost Risk:
{radiation_output.get(
    "risk_level",
    "N/A",
)}

Assessment Summary:
{radiation_output.get(
    "explanation",
    "Radiation frost conditions were assessed using "
    "temperature, wind, and cloud-cover information.",
)}
"""

        return {
            "final_response":
                final_response
        }

    # Soil Assessment

    if soil_output is not None:

        final_response = f"""
SOIL CONDITION ASSESSMENT

{metadata_text}

Soil Temperature:
{soil_output["soil_temperature"]:.1f} °C

Soil Risk:
{soil_output["soil_risk"].title()}

Assessment Summary:
{soil_output["explanation"]}
"""

        return {
            "final_response":
                final_response
        }

    # Weather-only Assessment

    if weather_output is not None:

        frost_probability = (
            weather_output[
                "frost_probability"
            ]
            * 100
        )

        risk_level = (
            _get_risk_level(
                frost_probability
            )
        )

        frost_prediction = (
            _get_frost_prediction_label(
                weather_output
            )
        )

        final_response = f"""
WEATHER-BASED FROST ASSESSMENT

{metadata_text}

Model Used:
Random Forest Frost Prediction Model

Frost Probability:
{frost_probability:.1f}%

Risk Level:
{risk_level}

Predicted Frost Event:
{frost_prediction}

Assessment Summary:
The trained Random Forest model estimates a
{frost_probability:.1f}% probability of frost for the evaluated
weather conditions.
"""

        return {
            "final_response":
                final_response
        }

    # Defensive final fallback

    final_response = """
No relevant frost-analysis output was produced.

Please rephrase the request or ask about frost prediction,
soil conditions, radiation frost, or frost-protection decisions.
"""

    return {
        "final_response":
            final_response
    }


# The four route_after_* functions below gate LangGraph's conditional
# edges. Which branches are actually reachable in each of them is
# constrained by LLMRouterAgent.INTENT_AGENT_MAP, which only ever
# produces the fixed combinations [], [WeatherAgent, SoilAgent],
# [SoilAgent], [RadiationFrostAgent], or
# [WeatherAgent, SoilAgent, RadiationFrostAgent, PlannerAgent]. Any
# candidate that can never appear alone (e.g. PlannerAgent without the
# other three) is omitted from that function's candidate list rather
# than carried as dead code.


def _route_to_first_required_agent(
    state: FrostState,
    candidates: list[tuple[str, str]],
) -> str:

    required_agents = state.get(
        "required_agents",
        [],
    )

    for agent_name, destination in candidates:

        if agent_name in required_agents:

            return destination

    return "response"


def route_after_selection(
    state: FrostState,
) -> str:

    return _route_to_first_required_agent(
        state,
        [
            ("WeatherAgent", "weather"),
            ("SoilAgent", "soil"),
            ("RadiationFrostAgent", "radiation"),
        ],
    )


def route_after_weather(
    state: FrostState,
) -> str:

    return _route_to_first_required_agent(
        state,
        [
            ("SoilAgent", "soil"),
        ],
    )


def route_after_soil(
    state: FrostState,
) -> str:

    return _route_to_first_required_agent(
        state,
        [
            ("RadiationFrostAgent", "radiation"),
        ],
    )


def route_after_radiation(
    state: FrostState,
) -> str:

    return _route_to_first_required_agent(
        state,
        [
            ("PlannerAgent", "planner"),
        ],
    )


graph = StateGraph(
    FrostState
)


graph.add_node(
    "query_understanding",
    query_understanding_node,
)

graph.add_node(
    "agent_selection",
    agent_selection_node,
)

graph.add_node(
    "weather",
    weather_node,
)

graph.add_node(
    "soil",
    soil_node,
)

graph.add_node(
    "radiation",
    radiation_node,
)

graph.add_node(
    "planner",
    planner_node,
)

graph.add_node(
    "response",
    response_node,
)


graph.set_entry_point(
    "query_understanding"
)


graph.add_edge(
    "query_understanding",
    "agent_selection",
)


graph.add_conditional_edges(
    "agent_selection",
    route_after_selection,
    {
        "weather":
            "weather",

        "soil":
            "soil",

        "radiation":
            "radiation",

        "planner":
            "planner",

        "response":
            "response",
    },
)


graph.add_conditional_edges(
    "weather",
    route_after_weather,
    {
        "soil":
            "soil",

        "radiation":
            "radiation",

        "planner":
            "planner",

        "response":
            "response",
    },
)


graph.add_conditional_edges(
    "soil",
    route_after_soil,
    {
        "radiation":
            "radiation",

        "planner":
            "planner",

        "response":
            "response",
    },
)


graph.add_conditional_edges(
    "radiation",
    route_after_radiation,
    {
        "planner":
            "planner",

        "response":
            "response",
    },
)


graph.add_edge(
    "planner",
    "response",
)


graph.add_edge(
    "response",
    END,
)


frost_graph = graph.compile()