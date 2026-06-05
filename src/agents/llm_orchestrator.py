# src/agents/llm_orchestrator.py
import json
import requests


class LLMOrchestrator:

    def __init__(
        self,
        model_name="llama3:latest"
    ):

        self.model_name = model_name

        self.url = (
            "http://localhost:11434/api/generate"
        )

    def explain(
        self,
        weather_output,
        soil_output,
        planner_output
    ):

        prompt = f"""
You are a frost-risk decision support assistant.

Use ONLY the information provided.

Weather Agent Output:
{json.dumps(weather_output, indent=2)}

Soil Agent Output:
{json.dumps(soil_output, indent=2)}

Planner Agent Output:
{json.dumps(planner_output, indent=2)}

Generate a short report with the following sections:

1. Weather Assessment
2. Soil Assessment
3. Recommended Action
4. Final Summary

Keep the report concise, professional, and easy to understand.
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:

            response = requests.post(
                self.url,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            return result.get(
                "response",
                self._fallback(
                    weather_output,
                    soil_output,
                    planner_output
                )
            )

        except Exception:

            return self._fallback(
                weather_output,
                soil_output,
                planner_output
            )

    def _fallback(
        self,
        weather_output,
        soil_output,
        planner_output
    ):

        return f"""
Weather Assessment
------------------
Frost probability: {weather_output['frost_probability']:.3f}
Predicted frost event: {weather_output['frost_prediction']}

Soil Assessment
---------------
Soil risk level: {soil_output['soil_risk']}
Soil temperature: {soil_output['soil_temperature']} °C

Recommended Action
------------------
{planner_output['recommended_action']}

Final Summary
-------------
The multi-agent system combined weather prediction,
soil assessment, and planning logic to generate
an operational recommendation.
"""