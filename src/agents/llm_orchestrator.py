# src/agents/llm_orchestrator.py

from __future__ import annotations

import re

import requests

from src.config import (
    LLM_MODEL_NAME,
    LLM_TIMEOUT_SECONDS,
)


class LLMOrchestrator:
    """
    Natural-language explanation layer for the frost-risk decision-support system.

    The LLM only explains already-computed WeatherAgent/SoilAgent/RadiationFrostAgent/
    PlannerAgent results; it never predicts, reclassifies, or overrides them. Generated
    text is validated against those authoritative outputs before being returned, and a
    deterministic grounded report is returned instead if validation fails.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        timeout_seconds: int = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.url = "http://localhost:11434/api/generate"

    @staticmethod
    def _extract_evidence(
        weather_output,
        soil_output,
        radiation_output,
        planner_output,
    ) -> dict:
        """Pull the authoritative fields shared by the prompt and the fallback report."""

        return {
            "frost_probability": float(weather_output.get("frost_probability", 0.0)) * 100,
            "frost_prediction": "Yes" if weather_output.get("frost_prediction") == 1 else "No",
            "soil_temperature": soil_output.get("soil_temperature"),
            "soil_risk": str(soil_output.get("soil_risk", "UNKNOWN")).upper(),
            "radiation_risk": str(radiation_output.get("risk_level", "UNKNOWN")).upper(),
            "air_temperature": radiation_output.get("air_temperature"),
            "soil_surface_temperature": radiation_output.get("soil_surface_temperature"),
            "wind_speed": radiation_output.get("wind_speed"),
            "cloud_cover": radiation_output.get("cloud_cover"),
            "priority": str(planner_output.get("priority", "UNKNOWN")).upper(),
            "recommended_action": str(
                planner_output.get("recommended_action", "No recommendation available.")
            ),
            "decision_basis": str(planner_output.get("decision_basis", "")),
            "evidence_status": str(planner_output.get("evidence_status", "UNKNOWN")).upper(),
        }

    def explain(
        self,
        weather_output,
        soil_output,
        radiation_output,
        planner_output,
    ) -> str:

        evidence = self._extract_evidence(
            weather_output, soil_output, radiation_output, planner_output
        )

        frost_probability = evidence["frost_probability"]
        frost_prediction = evidence["frost_prediction"]
        soil_temperature = evidence["soil_temperature"]
        soil_risk = evidence["soil_risk"]
        radiation_risk = evidence["radiation_risk"]
        air_temperature = evidence["air_temperature"]
        wind_speed = evidence["wind_speed"]
        cloud_cover = evidence["cloud_cover"]
        priority = evidence["priority"]
        recommended_action = evidence["recommended_action"]
        decision_basis = evidence["decision_basis"]
        evidence_status = evidence["evidence_status"]

        prompt = f"""
You are the explanation layer of a frost-risk decision-support system.

The prediction, specialist assessments, and operational decision have
already been computed.

You are NOT allowed to recalculate, reinterpret, reclassify, override,
or change them.

AUTHORITATIVE EVIDENCE

WeatherAgent:
- Frost probability: {frost_probability:.1f}%
- Frost predicted: {frost_prediction}
- Minimum air temperature: {self._format_temperature(air_temperature)}

SoilAgent:
- Soil-surface temperature at 0 cm: {self._format_temperature(soil_temperature)}
- Soil risk: {soil_risk}

RadiationFrostAgent:
- Radiation frost risk: {radiation_risk}
- Wind speed: {self._format_wind(wind_speed)}
- Cloud cover: {self._format_cloud(cloud_cover)}

PlannerAgent:
- Evidence status: {evidence_status}
- Priority: {priority}
- Recommended action: {recommended_action}
- Decision basis: {decision_basis}

Write a concise professional frost decision-support explanation.

Use exactly these sections:

Weather Assessment
Soil Assessment
Radiation Frost Assessment
Recommended Action
Final Summary

STRICT RULES

1. Use only the authoritative evidence above.

2. Do not change any numerical value.

3. Do not change:
   - Soil risk
   - Radiation frost risk
   - Planner priority
   - Planner recommendation
   - Frost prediction

4. Do not create your own LOW, MEDIUM, or HIGH classifications.

5. Do not describe cloud cover using qualitative terms such as
   "minimal", "moderate", "high", "heavy", or "low".
   Report the supplied percentage only.

6. Do not describe wind speed using qualitative terms such as
   "weak", "strong", "moderate", "calm", or "high".
   Report the supplied value only.

7. Do not claim that forecast values are physical sensor measurements.

8. Do not invent weather mechanisms or conditions that are not stated
   in the evidence.

9. Explain why the PlannerAgent recommendation follows from the
   specialist-agent evidence.

10. Do not change the recommendation even if you personally disagree.

11. Keep the report concise.

12. Do not add any recommendation beyond the PlannerAgent recommendation.
""".strip()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 220,
            },
            "keep_alive": "5m",
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()

            result = response.json()

            generated_response = result.get("response")

            if generated_response and generated_response.strip():
                generated_response = generated_response.strip()

                is_valid, reason = self._validate_generated_response(
                    generated_response=generated_response,
                    soil_risk=soil_risk,
                    radiation_risk=radiation_risk,
                    priority=priority,
                    recommended_action=recommended_action,
                    frost_prediction=frost_prediction,
                    cloud_cover=cloud_cover,
                )

                if is_valid:
                    return generated_response

                print("\nLLM explanation rejected by grounding validation.")
                print(f"Reason: {reason}")
                print("Using deterministic grounded fallback.")

        except requests.RequestException as error:
            print("\nLLM explanation unavailable. Using deterministic fallback.")
            print(f"Reason: {error}")

        except Exception as error:
            print("\nUnexpected LLM orchestrator error. Using deterministic fallback.")
            print(f"Reason: {error}")

        return self._fallback(
            weather_output,
            soil_output,
            radiation_output,
            planner_output,
        )

    def _validate_generated_response(
        self,
        generated_response: str,
        soil_risk: str,
        radiation_risk: str,
        priority: str,
        recommended_action: str,
        frost_prediction: str,
        cloud_cover,
    ) -> tuple[bool, str]:

        """
        Validate that the generated explanation does not contradict
        the authoritative structured outputs.

        This is intentionally conservative.

        The purpose is not to prove that every sentence is scientifically
        perfect. The purpose is to reject obvious contradictions between
        generated language and authoritative agent outputs.
        """

        text = generated_response.lower()

        # Required categorical evidence must appear
        if soil_risk.lower() not in text:

            return (
                False,
                (
                    "Generated explanation does not preserve "
                    f"SoilAgent risk classification: {soil_risk}."
                ),
            )

        if radiation_risk.lower() not in text:

            return (
                False,
                (
                    "Generated explanation does not preserve "
                    "RadiationFrostAgent risk classification: "
                    f"{radiation_risk}."
                ),
            )

        if priority.lower() not in text:

            return (
                False,
                (
                    "Generated explanation does not preserve "
                    f"PlannerAgent priority: {priority}."
                ),
            )

        # Recommendation must be represented
        normalized_action = self._normalize_text(
            recommended_action
        )

        normalized_generated = self._normalize_text(
            generated_response
        )

        important_action_terms = self._important_action_terms(
            normalized_action
        )

        for term in important_action_terms:

            if term not in normalized_generated:

                return (
                    False,
                    (
                        "Generated explanation does not preserve "
                        "the PlannerAgent recommendation."
                    ),
                )

        # Frost prediction contradiction check
        if frost_prediction == "No":

            contradiction_patterns = [
                r"\bfrost is predicted\b",
                r"\bfrost will occur\b",
                r"\bfrost is expected\b",
                r"\bfrost event is expected\b",
            ]

            for pattern in contradiction_patterns:

                if re.search(
                    pattern,
                    text,
                ):

                    return (
                        False,
                        (
                            "Generated explanation contradicts "
                            "WeatherAgent's NO-FROST prediction."
                        ),
                    )

        elif frost_prediction == "Yes":

            contradiction_patterns = [
                r"\bno frost is predicted\b",
                r"\bfrost is not predicted\b",
                r"\bno frost will occur\b",
            ]

            for pattern in contradiction_patterns:

                if re.search(
                    pattern,
                    text,
                ):

                    return (
                        False,
                        (
                            "Generated explanation contradicts "
                            "WeatherAgent's FROST prediction."
                        ),
                    )

        # Risk-classification contradiction checks
        soil_section = self._extract_section(
            generated_response,
            "soil assessment",
            "radiation frost assessment",
        )

        if soil_section:

            if self._contains_conflicting_risk(
                soil_section,
                expected_risk=soil_risk,
            ):

                return (
                    False,
                    (
                        "Generated Soil Assessment contains a "
                        "risk classification that conflicts with "
                        f"SoilAgent: {soil_risk}."
                    ),
                )

        radiation_section = self._extract_section(
            generated_response,
            "radiation frost assessment",
            "recommended action",
        )

        if radiation_section:

            if self._contains_conflicting_risk(
                radiation_section,
                expected_risk=radiation_risk,
            ):

                return (
                    False,
                    (
                        "Generated Radiation Frost Assessment "
                        "contains a risk classification that conflicts "
                        "with RadiationFrostAgent: "
                        f"{radiation_risk}."
                    ),
                )

        # Cloud-cover qualitative reinterpretation guard
        if cloud_cover is not None:

            cloud_patterns = [
                r"minimal cloud cover",
                r"low cloud cover",
                r"moderate cloud cover",
                r"high cloud cover",
                r"heavy cloud cover",
            ]

            for pattern in cloud_patterns:

                if re.search(
                    pattern,
                    text,
                ):

                    return (
                        False,
                        (
                            "Generated explanation qualitatively "
                            "reclassified cloud cover instead of "
                            "preserving the supplied forecast value."
                        ),
                    )

        return (
            True,
            "Grounding validation passed.",
        )

    @staticmethod
    def _contains_conflicting_risk(
        section: str,
        expected_risk: str,
    ) -> bool:

        section_lower = section.lower()

        expected = expected_risk.lower()

        possible_risks = {
            "low",
            "medium",
            "high",
        }

        conflicting = (
            possible_risks
            - {expected}
        )

        for risk in conflicting:

            patterns = [
                rf"\b{risk}\s+risk\b",
                rf"\brisk\s+is\s+{risk}\b",
                rf"\brisk\s+level\s+is\s+{risk}\b",
            ]

            for pattern in patterns:

                if re.search(
                    pattern,
                    section_lower,
                ):

                    return True

        return False

    @staticmethod
    def _extract_section(
        text: str,
        start_heading: str,
        end_heading: str,
    ) -> str:

        lower_text = text.lower()

        start_index = lower_text.find(
            start_heading
        )

        if start_index == -1:
            return ""

        end_index = lower_text.find(
            end_heading,
            start_index + len(start_heading),
        )

        if end_index == -1:

            return text[
                start_index:
            ]

        return text[
            start_index:
            end_index
        ]

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _important_action_terms(
        normalized_action: str,
    ) -> list[str]:

        """
        Extract a few important concepts from the PlannerAgent action.

        This avoids requiring an exact sentence match while still
        checking that the recommendation was preserved.
        """

        if (
            "activate frost protection"
            in normalized_action
        ):

            return [
                "activate",
                "frost protection",
            ]

        if (
            "monitor conditions closely"
            in normalized_action
        ):

            return [
                "monitor",
                "frost protection",
            ]

        if (
            "no immediate action required"
            in normalized_action
        ):

            return [
                "no",
                "action",
            ]

        return []

    @staticmethod
    def _format_temperature(
        value,
    ) -> str:

        if value is None:
            return "N/A"

        return f"{value:.1f} °C"

    @staticmethod
    def _format_wind(
        value,
    ) -> str:

        if value is None:
            return "N/A"

        return f"{value:.1f} km/h"

    @staticmethod
    def _format_cloud(
        value,
    ) -> str:

        if value is None:
            return "N/A"

        return f"{value:.1f}%"

    def _fallback(
        self,
        weather_output,
        soil_output,
        radiation_output,
        planner_output,
    ) -> str:

        evidence = self._extract_evidence(
            weather_output, soil_output, radiation_output, planner_output
        )

        frost_probability = evidence["frost_probability"]
        frost_prediction = evidence["frost_prediction"]
        soil_temperature = evidence["soil_temperature"]
        soil_risk = evidence["soil_risk"]
        radiation_risk = evidence["radiation_risk"]
        air_temperature = evidence["air_temperature"]
        soil_surface_temperature = evidence["soil_surface_temperature"]
        wind_speed = evidence["wind_speed"]
        cloud_cover = evidence["cloud_cover"]
        priority = evidence["priority"]
        recommended_action = evidence["recommended_action"]
        decision_basis = evidence["decision_basis"]
        evidence_status = evidence["evidence_status"]

        summary = (
            f"WeatherAgent estimates a frost probability of "
            f"{frost_probability:.1f}% with frost prediction "
            f"{frost_prediction}. "
            f"SoilAgent reports {soil_risk} soil risk, and "
            f"RadiationFrostAgent reports {radiation_risk} "
            f"radiation-frost risk. "
            f"The PlannerAgent therefore assigns {priority} "
            f"priority and recommends: {recommended_action}"
        )

        return f"""
Weather Assessment
------------------
Frost Probability: {frost_probability:.1f}%
Predicted Frost Event: {frost_prediction}
Minimum Air Temperature: {self._format_temperature(air_temperature)}

Soil Assessment
---------------
Soil-Surface Temperature (0 cm): {self._format_temperature(soil_temperature)}
Soil Risk: {soil_risk}

Radiation Frost Assessment
--------------------------
Radiation Frost Risk: {radiation_risk}
Minimum Air Temperature: {self._format_temperature(air_temperature)}
Soil-Surface Temperature (0 cm): {self._format_temperature(soil_surface_temperature)}
Average Wind Speed: {self._format_wind(wind_speed)}
Average Cloud Cover: {self._format_cloud(cloud_cover)}

Recommended Action
------------------
Evidence Status: {evidence_status}
Priority: {priority}
Action: {recommended_action}

Decision Basis:
{decision_basis}

Final Summary
-------------
{summary}
""".strip()