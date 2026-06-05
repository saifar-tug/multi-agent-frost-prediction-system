# src/main.py
from agents.weather_agent import WeatherAgent
from agents.soil_agent import SoilAgent
from agents.planner_agent import PlannerAgent
from agents.llm_orchestrator import LLMOrchestrator

import joblib
import json


# ------------------------------------------
# DEMO CONFIGURATION
# ------------------------------------------

DEMO_MODE = "frost"
# DEMO_MODE = "recent"


def print_section(title):

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def load_demo_sample():

    if DEMO_MODE == "frost":

        print("\nRunning Historical Frost Scenario")

        return joblib.load(
            "models/sample_row_frost.pkl"
        )

    elif DEMO_MODE == "recent":

        print("\nRunning Latest Available Observations")

        return joblib.load(
            "models/sample_row_recent.pkl"
        )

    else:

        raise ValueError(
            f"Unknown DEMO_MODE: {DEMO_MODE}"
        )


def main():

    print("\n" + "=" * 60)
    print("MULTI-AGENT FROST PREDICTION SYSTEM")
    print("=" * 60)

    # Load demonstration sample

    sample_weather_data = load_demo_sample()

    # Init the agents

    weather_agent = WeatherAgent()

    soil_agent = SoilAgent()

    planner_agent = PlannerAgent()

    llm_orchestrator = LLMOrchestrator(
        model_name="llama3:latest"
    )

    # --------------------------------------
    # Weather Agent
    # --------------------------------------

    weather_output = weather_agent.predict(
        sample_weather_data
    )

    print_section(
        "Weather Agent Assessment"
    )

    print(
        json.dumps(
            weather_output,
            indent=2
        )
    )

    # --------------------------------------
    # Soil Agent
    # --------------------------------------

    soil_output = soil_agent.assess(
        sample_weather_data
    )

    print_section(
        "Soil Agent Assessment"
    )

    print(
        json.dumps(
            soil_output,
            indent=2
        )
    )

    # --------------------------------------
    # Planner Agent
    # --------------------------------------

    planner_output = planner_agent.plan(
        weather_output,
        soil_output
    )

    print_section(
        "Planner Agent Recommendation"
    )

    print(
        json.dumps(
            planner_output,
            indent=2
        )
    )

    # --------------------------------------
    # LLM Orchestrator
    # --------------------------------------

    explanation = llm_orchestrator.explain(
        weather_output,
        soil_output,
        planner_output
    )

    print_section(
        "AI Decision Support Summary"
    )

    print(explanation)

    print("\n" + "=" * 60)
    print("MULTI-AGENT ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()