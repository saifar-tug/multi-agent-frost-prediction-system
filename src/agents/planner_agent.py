# src/agents/planner_agent.py

class PlannerAgent:

    def plan(
        self,
        weather_output,
        soil_output
    ):

        frost_probability = (
            weather_output["frost_probability"]
        )

        soil_risk = (
            soil_output["soil_risk"]
        )

        if (
            frost_probability >= 0.7
            and soil_risk == "high"
        ):

            action = (
                "Activate frost protection."
            )

            priority = "HIGH"

        elif (
            frost_probability >= 0.5
        ):

            action = (
                "Monitor conditions closely."
            )

            priority = "MEDIUM"

        else:

            action = (
                "No immediate action required."
            )

            priority = "LOW"

        return {
            "agent": "PlannerAgent",
            "priority": priority,
            "recommended_action": action
        }