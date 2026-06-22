import json
import re
import requests


class LLMRouterAgent:

    def __init__(
        self,
        model_name="llama3:latest",
        ollama_url="http://localhost:11434/api/generate"
    ):

        self.model_name = model_name
        self.ollama_url = ollama_url

    def select_agents(
        self,
        question: str
    ) -> list[str]:

        prompt = f"""
You are the routing agent of a Multi-Agent Frost Prediction and Decision Support System.

Available agents:

WeatherAgent
- frost prediction
- frost probability
- weather forecast

SoilAgent
- soil temperature
- soil condition
- crop vulnerability

RadiationFrostAgent
- radiation frost assessment
- cloud cover influence
- wind influence
- physical frost formation

PlannerAgent
- recommendations
- protection measures
- operational decisions

Routing Rules:

1. Soil-related questions:
["SoilAgent"]

2. Frost prediction questions:
["WeatherAgent","SoilAgent"]

3. Action or recommendation questions:
["WeatherAgent","SoilAgent","RadiationFrostAgent","PlannerAgent"]

Return ONLY valid JSON.

Example:

{{"agents":["SoilAgent"]}}

Question:
{question}
"""

        try:

            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0
                },
                timeout=60
            )

            response.raise_for_status()

            llm_response = response.json()["response"]

            json_match = re.search(
                r"\{[\s\S]*\}",
                llm_response
            )

            if not json_match:

                raise ValueError(
                    "No JSON found in router output."
                )

            parsed = json.loads(
                json_match.group()
            )

            agents = parsed.get(
                "agents",
                []
            )

            # --------------------------------------------------
            # Architecture Enforcement Rules
            # --------------------------------------------------

            question_lower = question.lower()

            # Frost prediction should use both
            # WeatherAgent and SoilAgent

            if any(
                keyword in question_lower
                for keyword in [
                    "frost",
                    "forecast",
                    "tomorrow",
                    "tonight",
                    "weather",
                    "temperature"
                ]
            ):

                if "WeatherAgent" not in agents:

                    agents.append(
                        "WeatherAgent"
                    )

                if "SoilAgent" not in agents:

                    agents.append(
                        "SoilAgent"
                    )

            # Planner requires all upstream agents

            if "PlannerAgent" in agents:

                agents = [
                    "WeatherAgent",
                    "SoilAgent",
                    "RadiationFrostAgent",
                    "PlannerAgent"
                ]

            # Remove duplicates while preserving order

            agents = list(
                dict.fromkeys(
                    agents
                )
            )

            return agents

        except Exception as e:

            print(
                f"\nRouter fallback activated: {e}"
            )

            return [
                "WeatherAgent",
                "SoilAgent"
            ]