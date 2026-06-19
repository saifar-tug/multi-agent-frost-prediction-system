# src/router/llm_router.py


class LLMRouter:

    def route(
        self,
        user_question
    ):

        question = (
            user_question
            .lower()
            .strip()
        )

        # Soil-related questions

        if any(
            keyword in question
            for keyword in [
                "soil",
                "ground",
                "soil temperature",
                "soil risk"
            ]
        ):

            return [
                "SoilAgent"
            ]

        # Action / recommendation questions

        if any(
            keyword in question
            for keyword in [
                "action",
                "recommend",
                "recommendation",
                "protect",
                "protection",
                "heater",
                "fan",
                "should i",
                "what should i do"
            ]
        ):

            return [
                "WeatherAgent",
                "SoilAgent",
                "PlannerAgent"
            ]

        # Frost / weather questions

        if any(
            keyword in question
            for keyword in [
                "frost",
                "weather",
                "forecast",
                "temperature",
                "tomorrow",
                "tonight"
            ]
        ):

            return [
                "WeatherAgent",
                "SoilAgent"
            ]

        # Default

        return [
            "WeatherAgent",
            "SoilAgent"
        ]