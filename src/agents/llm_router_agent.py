# src/agents/llm_router_agent.py

from __future__ import annotations

from typing import Any


class LLMRouterAgent:
    """
    Agent-selection policy for the frost decision-support system.

    Receives structured intent from QueryUnderstandingAgent and
    deterministically decides which domain agents must run; it never
    activates frost agents as a fallback for unknown or out-of-domain
    questions.
    """

    INTENT_AGENT_MAP = {

        "frost_prediction": [
            "WeatherAgent",
            "SoilAgent",
        ],

        "soil_assessment": [
            "SoilAgent",
        ],

        "radiation_frost_assessment": [
            "RadiationFrostAgent",
        ],

        "protection_decision": [
            "WeatherAgent",
            "SoilAgent",
            "RadiationFrostAgent",
            "PlannerAgent",
        ],

        "frost_explanation": [],

        "greeting": [],

        "acknowledgement": [],

        "out_of_scope": [],

        "unknown": [],
    }

    def select_agents(
        self,
        query_understanding: dict[str, Any],
    ) -> list[str]:
        """
        Select domain agents from structured query understanding.

        Parameters
        ----------
        query_understanding:
            Output produced by QueryUnderstandingAgent.

        Returns
        -------
        list[str]
            Ordered list of agents required for the current request.
        """

        domain = query_understanding.get(
            "domain",
            "unknown",
        )

        intent = query_understanding.get(
            "intent",
            "unknown",
        )

        if domain in {
            "out_of_scope",
            "unknown",
        }:

            return []

        if domain == "general_conversation":

            return []

        # Only frost-domain requests may activate frost-analysis agents
        if domain != "frost":

            return []

        required_agents = (
            self.INTENT_AGENT_MAP.get(
                intent,
                [],
            )
        )

        return list(
            required_agents
        )