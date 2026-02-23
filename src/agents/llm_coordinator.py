# src/agents/llm_coordinator.py

class LLMCoordinator:
    def __init__(self):
        pass

    def generate_summary(self, weather_output, soil_output, planner_output):
        """
        Generate structured explanation.
        """

        summary = (
            f"Frost Probability: {weather_output['frost_probability']:.2f}. "
            f"Crop Vulnerability: {soil_output['vulnerability_score']:.2f}. "
            f"Recommended Action: {planner_output['recommended_action']}."
        )

        return summary