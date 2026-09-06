# src/agents/planner_agent.py


class PlannerAgent:

    def plan(
        self,
        weather_output,
        soil_output,
        radiation_output,
    ):

        frost_probability = float(
            weather_output["frost_probability"]
        )

        soil_risk = str(
            soil_output.get(
                "soil_risk",
                "unknown",
            )
        ).upper()

        radiation_risk = str(
            radiation_output.get(
                "risk_level",
                "UNKNOWN",
            )
        ).upper()

        # WeatherAgent provides the primary learned statistical frost estimate;
        # SoilAgent and RadiationFrostAgent provide complementary physical evidence.
        # Strong agreement produces a HIGH-priority recommendation. Conflicting or
        # moderately concerning evidence produces a MEDIUM-priority monitoring decision
        # rather than allowing one specialist to override all other evidence.

        high_support_count = sum(
            [
                soil_risk == "HIGH",
                radiation_risk == "HIGH",
            ]
        )

        concerning_support_count = sum(
            [
                soil_risk in {
                    "MEDIUM",
                    "HIGH",
                },
                radiation_risk in {
                    "MEDIUM",
                    "HIGH",
                },
            ]
        )

        # HIGH: strong RF evidence plus at least one strong physical warning, or
        # moderate RF evidence with both physical specialists reporting HIGH risk.

        if (
            (
                frost_probability >= 0.70
                and high_support_count >= 1
            )
            or
            (
                frost_probability >= 0.50
                and high_support_count == 2
            )
        ):

            priority = "HIGH"

            action = (
                "Activate frost protection."
            )

            decision_basis = (
                "The learned frost model and physical "
                "risk assessments provide strong combined "
                "evidence of frost risk."
            )

        # MEDIUM: moderate/high RF probability without sufficient physical agreement,
        # or physical specialists show concerning conditions despite a lower RF score.

        elif (
            frost_probability >= 0.50
            or concerning_support_count >= 1
        ):

            priority = "MEDIUM"

            action = (
                "Monitor conditions closely and prepare "
                "frost-protection measures."
            )

            decision_basis = (
                "At least one specialist indicates "
                "meaningful frost risk, but the combined "
                "evidence is not strong enough for an "
                "immediate high-priority intervention."
            )

        else:

            priority = "LOW"

            action = (
                "No immediate action required."
            )

            decision_basis = (
                "The learned frost probability is low and "
                "the physical assessments do not indicate "
                "meaningful frost risk."
            )

        if (
            frost_probability >= 0.50
            and soil_risk in {"MEDIUM", "HIGH"}
            and radiation_risk in {"MEDIUM", "HIGH"}
        ):

            evidence_status = "AGREEMENT"

        elif (
            frost_probability < 0.50
            and soil_risk == "LOW"
            and radiation_risk == "LOW"
        ):

            evidence_status = "AGREEMENT"

        else:

            evidence_status = "MIXED"

        return {
            "agent":
                "PlannerAgent",

            "priority":
                priority,

            "recommended_action":
                action,

            "decision_basis":
                decision_basis,

            "evidence_status":
                evidence_status,

            "evidence": {
                "frost_probability":
                    frost_probability,

                "soil_risk":
                    soil_risk,

                "radiation_frost_risk":
                    radiation_risk,
            },
        }