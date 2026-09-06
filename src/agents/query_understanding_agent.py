# src/agents/query_understanding_agent.py

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import requests

from src.config import (
    LLM_MODEL_NAME,
    LLM_TIMEOUT_SECONDS,
)


@dataclass
class QueryUnderstandingResult:
    """Structured interpretation of a user question."""

    domain: str
    intent: str
    entities: dict[str, Any]
    confidence: float
    reasoning: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryUnderstandingAgent:
    """
    Interpret a user question before any domain agent is activated.

    Architecture:
    1. Deterministic classification handles clear, high-confidence cases.
    2. The LLM handles semantically ambiguous natural-language requests.
    3. LLM output is validated and normalized before it can be used
       for downstream routing.
    4. If the LLM fails, the request remains safely unknown.

    This prevents malformed or unrelated LLM output from accidentally
    activating frost-analysis agents.
    """

    VALID_DOMAINS = {
        "frost",
        "general_conversation",
        "out_of_scope",
        "unknown",
    }

    VALID_INTENTS = {
        "frost_prediction",
        "soil_assessment",
        "radiation_frost_assessment",
        "protection_decision",
        "frost_explanation",
        "greeting",
        "acknowledgement",
        "out_of_scope",
        "unknown",
    }

    INTENT_DOMAIN_MAP = {
        "frost_prediction": "frost",
        "soil_assessment": "frost",
        "radiation_frost_assessment": "frost",
        "protection_decision": "frost",
        "frost_explanation": "frost",
        "greeting": "general_conversation",
        "acknowledgement": "general_conversation",
        "out_of_scope": "out_of_scope",
    }

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        ollama_url: str = "http://localhost:11434/api/generate",
        timeout_seconds: int = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout_seconds = timeout_seconds

    def understand(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Return a structured interpretation of the user question.

        Returned fields:
        - domain
        - intent
        - entities
        - confidence
        - reasoning
        - source
        """

        cleaned_question = question.strip()

        if not cleaned_question:
            return QueryUnderstandingResult(
                domain="unknown",
                intent="unknown",
                entities={
                    "time_reference": None,
                    "location": None,
                    "topic": None,
                },
                confidence=1.0,
                reasoning="The question is empty.",
                source="deterministic",
            ).to_dict()

        # Deterministic rules run first; the LLM is only a fallback for requests
        # they can't confidently classify.
        deterministic_result = self._understand_deterministically(
            cleaned_question
        )

        if deterministic_result.domain != "unknown":
            return deterministic_result.to_dict()

        try:
            llm_result = self._understand_with_llm(
                cleaned_question
            )

            return llm_result.to_dict()

        except Exception as exc:
            return QueryUnderstandingResult(
                domain="unknown",
                intent="unknown",
                entities={
                    "time_reference": self._extract_time_reference(
                        cleaned_question.lower()
                    ),
                    "location": self._extract_location(
                        cleaned_question.lower()
                    ),
                    "topic": None,
                },
                confidence=0.0,
                reasoning=(
                    "The request could not be classified safely. "
                    f"LLM cause: {exc}"
                ),
                source="deterministic_fallback",
            ).to_dict()

    def _understand_with_llm(
        self,
        question: str,
    ) -> QueryUnderstandingResult:
        """Use the configured Ollama model for semantic interpretation."""

        prompt = self._build_prompt(
            question
        )

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 180,
                },
                "keep_alive": "5m",
            },
            timeout=self.timeout_seconds,
        )

        response.raise_for_status()

        payload = response.json()

        raw_response = payload.get(
            "response",
            "",
        )

        parsed = self._extract_json(
            raw_response
        )

        return self._validate_result(
            parsed,
            source="llm",
        )

    def _build_prompt(
        self,
        question: str,
    ) -> str:
        """Build the semantic-classification prompt."""

        return f"""
You are the query-understanding layer of a frost-risk
prediction and decision-support system.

Do NOT answer the user's question.

Your only task is to classify and interpret the request.

SUPPORTED FROST DOMAIN

The system supports:

- frost prediction
- frost-risk assessment
- soil or soil-surface frost assessment
- radiation-frost assessment
- frost formation explanation
- frost-protection decisions

General weather questions that do not request frost analysis are
outside the supported application domain.

VALID DOMAINS

- frost
- general_conversation
- out_of_scope
- unknown

VALID INTENTS

- frost_prediction
- soil_assessment
- radiation_frost_assessment
- protection_decision
- frost_explanation
- greeting
- acknowledgement
- out_of_scope
- unknown


CLASSIFICATION EXAMPLES

Question:
Will frost occur tomorrow?

Classification:
domain = frost
intent = frost_prediction


Question:
What is the soil condition?

Classification:
domain = frost
intent = soil_assessment


Question:
Could radiation frost occur tonight?

Classification:
domain = frost
intent = radiation_frost_assessment


Question:
Should I activate frost protection tonight?

Classification:
domain = frost
intent = protection_decision


Question:
Why does radiation frost occur?

Classification:
domain = frost
intent = frost_explanation


Question:
Do I need to worry about my plants freezing overnight?

Classification:
domain = frost
intent = protection_decision


Question:
Could the crops be damaged by freezing conditions tonight?

Classification:
domain = frost
intent = protection_decision


Question:
What is the temperature in Vienna?

Classification:
domain = out_of_scope
intent = out_of_scope


Question:
Will it rain tomorrow?

Classification:
domain = out_of_scope
intent = out_of_scope


Question:
What is the weather in Graz?

Classification:
domain = out_of_scope
intent = out_of_scope


Question:
What is the oil price today?

Classification:
domain = out_of_scope
intent = out_of_scope


Question:
Hello

Classification:
domain = general_conversation
intent = greeting


OUTPUT FORMAT

Return ONLY one valid JSON object.

Example structure:

{{
  "domain": "frost",
  "intent": "frost_prediction",
  "entities": {{
    "time_reference": "tomorrow",
    "location": "Graz, Austria",
    "topic": "frost prediction"
  }},
  "confidence": 0.95,
  "reasoning": "The user is asking whether frost will occur."
}}

If an entity is absent, use JSON null.

For example:

{{
  "domain": "general_conversation",
  "intent": "greeting",
  "entities": {{
    "time_reference": null,
    "location": null,
    "topic": null
  }},
  "confidence": 0.95,
  "reasoning": "The user is greeting the system."
}}


IMPORTANT RULES

1. A time word such as "today", "tonight", or "tomorrow" does
   NOT by itself make a request a frost request.

2. A location such as Graz or Vienna does NOT by itself make
   a request a frost request.

3. General temperature, rain, sunshine, wind, humidity, snow,
   or ordinary weather questions are outside the supported
   frost-analysis domain unless the user connects them to
   frost, freezing, crop frost damage, or frost risk.

4. Never classify ordinary weather requests as frost requests
   simply because weather information could theoretically be
   relevant to frost.

5. If the user asks whether frost or freezing conditions will
   occur, use frost_prediction.

6. If the user asks specifically about soil or ground frost
   conditions, use soil_assessment.

7. If the user asks specifically about radiation frost or
   radiative-cooling conditions, use
   radiation_frost_assessment.

8. If the user asks what action should be taken to protect
   crops, plants, or other frost-sensitive assets, use
   protection_decision.

9. If the user asks conceptually what frost is, why it occurs,
   or how it forms, use frost_explanation.

10. Use unknown only when the meaning genuinely cannot be
    determined.

11. Use JSON null when an entity is absent.

12. Never return the literal strings:
    "null",
    "None",
    "string or null",
    or "unknown"
    as substitutes for a missing entity.

13. time_reference examples:
    "today",
    "tonight",
    "tomorrow",
    "tomorrow night".

14. location examples:
    "Graz, Austria",
    "Vienna, Austria".

15. topic should briefly describe the subject of the request.

16. Return JSON only. Do not include Markdown, commentary,
    headings, or an answer to the user's actual question.


USER QUESTION

{question}
""".strip()

    @staticmethod
    def _extract_json(
        raw_response: str,
    ) -> dict[str, Any]:
        """Extract one JSON object from the LLM response."""

        match = re.search(
            r"\{[\s\S]*\}",
            raw_response,
        )

        if not match:
            raise ValueError(
                "No JSON object found in the LLM response."
            )

        try:
            parsed = json.loads(
                match.group()
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "The LLM returned invalid JSON."
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "The parsed LLM result is not a JSON object."
            )

        return parsed

    def _validate_result(
        self,
        parsed: dict[str, Any],
        source: str,
    ) -> QueryUnderstandingResult:
        """
        Validate and normalize an LLM classification.

        Validation includes:
        - allowed domains
        - allowed intents
        - domain/intent consistency
        - entity normalization
        - confidence normalization
        """

        domain = str(
            parsed.get(
                "domain",
                "unknown",
            )
        ).strip().lower()

        intent = str(
            parsed.get(
                "intent",
                "unknown",
            )
        ).strip().lower()

        if domain not in self.VALID_DOMAINS:
            domain = "unknown"

        if intent not in self.VALID_INTENTS:
            intent = "unknown"

        # Enforce semantic consistency (e.g. protection_decision must belong to
        # frost, greeting must belong to general_conversation) even if the LLM
        # paired the intent with an inconsistent domain.
        expected_domain = self.INTENT_DOMAIN_MAP.get(
            intent
        )

        if expected_domain is not None:
            domain = expected_domain

        entities = parsed.get(
            "entities",
            {},
        )

        if not isinstance(
            entities,
            dict,
        ):
            entities = {}

        normalized_entities = {
            "time_reference": self._normalize_entity(
                entities.get("time_reference")
            ),
            "location": self._normalize_location(
                entities.get("location")
            ),
            "topic": self._normalize_entity(
                entities.get("topic")
            ),
        }

        try:
            confidence = float(
                parsed.get(
                    "confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        reasoning = str(
            parsed.get(
                "reasoning",
                "No reasoning was provided.",
            )
        ).strip()

        if not reasoning:
            reasoning = (
                "No reasoning was provided."
            )

        return QueryUnderstandingResult(
            domain=domain,
            intent=intent,
            entities=normalized_entities,
            confidence=confidence,
            reasoning=reasoning,
            source=source,
        )

    @staticmethod
    def _normalize_entity(
        value: Any,
    ) -> Any:
        """Normalize missing or malformed entity values."""

        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            return value

        cleaned = value.strip()

        invalid_values = {
            "",
            "null",
            "none",
            "unknown",
            "string or null",
            "n/a",
        }

        if cleaned.lower() in invalid_values:
            return None

        return cleaned

    @staticmethod
    def _normalize_location(
        value: Any,
    ) -> str | None:
        """
        Normalize geographic location entities.

        Reject common non-geographic objects or environmental
        descriptions that an LLM may incorrectly classify as locations.
        """

        normalized = QueryUnderstandingAgent._normalize_entity(
            value
        )

        if normalized is None:
            return None

        if not isinstance(normalized, str):
            return None

        non_geographic_terms = {
            "vineyard",
            "field",
            "farm",
            "garden",
            "ground",
            "soil",
            "plants",
            "plant",
            "crops",
            "crop",
            "outside",
            "outdoors",
        }

        if normalized.lower() in non_geographic_terms:
            return None

        return normalized

    def _understand_deterministically(
        self,
        question: str,
    ) -> QueryUnderstandingResult:
        """
        Classify clear, high-confidence requests locally.

        If the request is not sufficiently clear, return unknown so
        that the semantic LLM layer can interpret it.
        """

        normalized = re.sub(
            r"\s+",
            " ",
            question.lower(),
        ).strip()

        entities = {
            "time_reference": self._extract_time_reference(
                normalized
            ),
            "location": self._extract_location(
                normalized
            ),
            "topic": None,
        }

        if self._is_greeting(
            normalized
        ):
            return QueryUnderstandingResult(
                domain="general_conversation",
                intent="greeting",
                entities=entities,
                confidence=0.99,
                reasoning=(
                    "A clear greeting was detected."
                ),
                source="deterministic",
            )

        if self._is_acknowledgement(
            normalized
        ):
            return QueryUnderstandingResult(
                domain="general_conversation",
                intent="acknowledgement",
                entities=entities,
                confidence=0.99,
                reasoning=(
                    "A clear acknowledgement was detected."
                ),
                source="deterministic",
            )

        if self._contains_protection_intent(
            normalized
        ):
            entities["topic"] = (
                "frost protection"
            )

            return QueryUnderstandingResult(
                domain="frost",
                intent="protection_decision",
                entities=entities,
                confidence=0.98,
                reasoning=(
                    "The question explicitly requests an "
                    "operational frost-protection decision."
                ),
                source="deterministic",
            )

        if self._contains_soil_intent(
            normalized
        ):
            entities["topic"] = (
                "soil frost condition"
            )

            return QueryUnderstandingResult(
                domain="frost",
                intent="soil_assessment",
                entities=entities,
                confidence=0.97,
                reasoning=(
                    "The question explicitly requests assessment "
                    "of soil or ground-level frost conditions."
                ),
                source="deterministic",
            )

        # Check explanation BEFORE radiation assessment so "Explain radiation
        # frost" classifies as frost_explanation, not radiation_frost_assessment.
        if self._contains_explanation_intent(
            normalized
        ):
            entities["topic"] = (
                "frost explanation"
            )

            return QueryUnderstandingResult(
                domain="frost",
                intent="frost_explanation",
                entities=entities,
                confidence=0.97,
                reasoning=(
                    "The question explicitly requests a "
                    "conceptual explanation of frost."
                ),
                source="deterministic",
            )

        if self._contains_radiation_intent(
            normalized
        ):
            entities["topic"] = (
                "radiation frost"
            )

            return QueryUnderstandingResult(
                domain="frost",
                intent="radiation_frost_assessment",
                entities=entities,
                confidence=0.97,
                reasoning=(
                    "The question explicitly requests assessment "
                    "of radiation-frost conditions."
                ),
                source="deterministic",
            )

        if self._contains_frost_prediction_intent(
            normalized
        ):
            entities["topic"] = (
                "frost prediction"
            )

            return QueryUnderstandingResult(
                domain="frost",
                intent="frost_prediction",
                entities=entities,
                confidence=0.98,
                reasoning=(
                    "The question explicitly requests a frost "
                    "prediction or frost-risk assessment."
                ),
                source="deterministic",
            )

        if self._is_general_weather_request(
            normalized
        ):
            entities["topic"] = (
                self._detect_weather_topic(
                    normalized
                )
            )

            return QueryUnderstandingResult(
                domain="out_of_scope",
                intent="out_of_scope",
                entities=entities,
                confidence=0.98,
                reasoning=(
                    "The question is a general weather request "
                    "rather than a frost-risk request."
                ),
                source="deterministic",
            )

        if self._is_out_of_scope(
            normalized
        ):
            entities["topic"] = (
                self._detect_out_of_scope_topic(
                    normalized
                )
            )

            return QueryUnderstandingResult(
                domain="out_of_scope",
                intent="out_of_scope",
                entities=entities,
                confidence=0.99,
                reasoning=(
                    "The question concerns a topic outside "
                    "the frost-risk decision-support domain."
                ),
                source="deterministic",
            )

        # Ambiguous request: the caller falls back to the LLM when domain is "unknown".
        return QueryUnderstandingResult(
            domain="unknown",
            intent="unknown",
            entities=entities,
            confidence=0.0,
            reasoning=(
                "The deterministic classifier could not "
                "confidently classify the request."
            ),
            source="deterministic",
        )

    @staticmethod
    def _is_greeting(
        question: str,
    ) -> bool:

        greetings = {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        }

        return question in greetings

    @staticmethod
    def _is_acknowledgement(
        question: str,
    ) -> bool:

        acknowledgements = {
            "ok",
            "okay",
            "thanks",
            "thank you",
            "understood",
            "got it",
        }

        return question in acknowledgements

    @staticmethod
    def _contains_protection_intent(
        question: str,
    ) -> bool:

        explicit_terms = (
            "activate frost protection",
            "frost protection",
            "protect from frost",
            "protect against frost",
            "frost prevention",
        )

        if any(
            term in question
            for term in explicit_terms
        ):
            return True

        action_terms = (
            "what should i do",
            "what should we do",
            "should i activate",
            "should we activate",
            "recommended action",
            "recommendation",
            "take action",
        )

        frost_context = (
            "frost" in question
            or "freezing" in question
            or "freeze" in question
        )

        return (
            frost_context
            and any(
                term in question
                for term in action_terms
            )
        )

    @staticmethod
    def _contains_soil_intent(
        question: str,
    ) -> bool:

        terms = (
            "soil temperature",
            "soil condition",
            "soil frost",
            "soil surface temperature",
            "soil-surface temperature",
            "ground temperature",
            "ground condition",
            "ground frost",
        )

        return any(
            term in question
            for term in terms
        )

    @staticmethod
    def _contains_radiation_intent(
        question: str,
    ) -> bool:

        explicit_terms = (
            "radiation frost",
            "radiative frost",
            "radiative cooling",
            "nighttime radiative cooling",
        )

        if any(
            term in question
            for term in explicit_terms
        ):
            return True

        frost_context = (
            "frost" in question
            or "freezing" in question
            or "freeze" in question
        )

        radiation_factors = (
            "cloud cover",
            "clear sky",
            "clear skies",
            "calm wind",
            "wind speed",
            "radiative cooling",
        )

        return (
            frost_context
            and any(
                term in question
                for term in radiation_factors
            )
        )

    @staticmethod
    def _contains_explanation_intent(
        question: str,
    ) -> bool:

        if "frost" not in question:
            return False

        explanation_terms = (
            "explain",
            "what is frost",
            "what is radiation frost",
            "what is radiative frost",
            "how does frost",
            "why does frost",
            "what causes frost",
            "how frost forms",
            "how does radiation frost",
            "why does radiation frost",
        )

        return any(
            term in question
            for term in explanation_terms
        )

    @staticmethod
    def _contains_frost_prediction_intent(
        question: str,
    ) -> bool:

        frost_terms = (
            "frost",
            "frost risk",
            "frosty",
            "freeze",
            "freezing",
            "below freezing",
        )

        return any(
            term in question
            for term in frost_terms
        )

    @staticmethod
    def _is_general_weather_request(
        question: str,
    ) -> bool:

        weather_terms = (
            "temperature",
            "weather",
            "rain",
            "raining",
            "precipitation",
            "sunny",
            "sunshine",
            "wind",
            "windy",
            "humidity",
            "cloud cover",
            "forecast",
            "snow",
        )

        return any(
            term in question
            for term in weather_terms
        )

    @staticmethod
    def _detect_weather_topic(
        question: str,
    ) -> str:

        topic_map = {
            "temperature": "temperature",
            "weather": "weather",
            "rain": "rain",
            "precipitation": "precipitation",
            "sunny": "sunshine",
            "sunshine": "sunshine",
            "wind": "wind",
            "humidity": "humidity",
            "cloud cover": "cloud cover",
            "forecast": "weather forecast",
            "snow": "snow",
        }

        for term, topic in topic_map.items():
            if term in question:
                return topic

        return "general weather"

    @staticmethod
    def _is_out_of_scope(
        question: str,
    ) -> bool:

        terms = (
            "oil price",
            "bitcoin",
            "cryptocurrency",
            "stock price",
            "stock market",
            "football",
            "soccer",
            "election",
            "president",
            "politics",
            "recipe",
            "movie",
            "music",
        )

        return any(
            term in question
            for term in terms
        )

    @staticmethod
    def _detect_out_of_scope_topic(
        question: str,
    ) -> str:

        topic_map = {
            "oil price": "oil price",
            "bitcoin": "cryptocurrency",
            "cryptocurrency": "cryptocurrency",
            "stock price": "stock market",
            "stock market": "stock market",
            "football": "football",
            "soccer": "football",
            "election": "politics",
            "president": "politics",
            "politics": "politics",
            "recipe": "cooking",
            "movie": "movies",
            "music": "music",
        }

        for term, topic in topic_map.items():
            if term in question:
                return topic

        return "unrelated topic"

    @staticmethod
    def _extract_time_reference(
        question: str,
    ) -> str | None:

        for term in (
            "tomorrow night",
            "tomorrow morning",
            "tomorrow evening",
            "this evening",
            "this morning",
            "overnight",
            "tomorrow",
            "tonight",
            "today",
        ):
            if term in question:
                return term

        return None

    @staticmethod
    def _extract_location(
        question: str,
    ) -> str | None:

        known_locations = {
            "graz": "Graz, Austria",
            "vienna": "Vienna, Austria",
            "wien": "Vienna, Austria",
            "austria": "Austria",
        }

        for term, location in known_locations.items():
            if term in question:
                return location

        return None