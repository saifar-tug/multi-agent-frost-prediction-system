# src/router/chat_interface.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "src")
)

from src.agents.weather_agent import WeatherAgent
from src.agents.soil_agent import SoilAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.llm_orchestrator import LLMOrchestrator

from src.router.llm_router import LLMRouter

import joblib
import json


class FrostAssistant:

    def __init__(self):

        self.router = LLMRouter()

        self.weather_agent = WeatherAgent()

        self.soil_agent = SoilAgent()

        self.planner_agent = PlannerAgent()

        self.llm = LLMOrchestrator(
            model_name="llama3:latest"
        )

        self.sample_data = joblib.load(
            "models/sample_row_frost.pkl"
        )

    def ask(self, question):

        selected_agents = self.router.route(
            question
        )

        print("\nActivated Agents:")
        print(selected_agents)

        weather_output = None
        soil_output = None
        planner_output = None

        if "WeatherAgent" in selected_agents:

            weather_output = (
                self.weather_agent.predict(
                    self.sample_data
                )
            )

        if "SoilAgent" in selected_agents:

            soil_output = (
                self.soil_agent.assess(
                    self.sample_data
                )
            )

        if "PlannerAgent" in selected_agents:

            if weather_output is None:

                weather_output = (
                    self.weather_agent.predict(
                        self.sample_data
                    )
                )

            if soil_output is None:

                soil_output = (
                    self.soil_agent.assess(
                        self.sample_data
                    )
                )

            planner_output = (
                self.planner_agent.plan(
                    weather_output,
                    soil_output
                )
            )

        if planner_output:

            return self.llm.explain(
                weather_output,
                soil_output,
                planner_output
            )

        response = {}

        if weather_output:

            response["weather"] = (
                weather_output
            )

        if soil_output:

            response["soil"] = (
                soil_output
            )

        return json.dumps(
            response,
            indent=2
        )


def main():

    assistant = FrostAssistant()

    print("\n" + "=" * 60)
    print("FROST ASSISTANT")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk a question ('exit' to quit): "
        )

        if question.lower() == "exit":

            break

        answer = assistant.ask(
            question
        )

        print("\nResponse:")
        print(answer)


if __name__ == "__main__":
    main()