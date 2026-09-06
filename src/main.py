# src/main.py

from __future__ import annotations
from typing import Any
import joblib

from src.data_pipeline.live_data_loader import (
    LiveWeatherDataError,
    build_feature_row,)
from src.graph.frost_graph import frost_graph


# Execution configuration
SYSTEM_MODE = "live"

DEMO_MODE = "frost"


def load_demo_sample() -> dict[str, Any]:
    """Load one saved historical demonstration record."""

    if DEMO_MODE == "frost":
        return joblib.load(
            "models/sample_row_frost.pkl"
        )

    if DEMO_MODE == "recent":
        return joblib.load(
            "models/sample_row_recent.pkl"
        )

    raise ValueError(
        f"Unknown DEMO_MODE: {DEMO_MODE}"
    )


def load_input_data() -> dict[str, Any]:
    """
    Load the input record for the selected execution mode.

    Historical mode:
        Loads a saved GeoSphere observation.

    Live mode:
        Retrieves tomorrow's Open-Meteo forecast and maps it to the
        GeoSphere-compatible model feature schema.
    """

    if SYSTEM_MODE == "historical":
        return load_demo_sample()

    if SYSTEM_MODE == "live":
        return build_feature_row()

    raise ValueError(
        f"Unknown SYSTEM_MODE: {SYSTEM_MODE}. "
        "Use 'historical' or 'live'."
    )


def print_system_header(
    sample_data: dict[str, Any],
) -> None:
    """Print execution metadata for the selected data source."""

    print("\n")
    print("=" * 60)
    print("MULTI-AGENT FROST PREDICTION SYSTEM")
    print("=" * 60)

    if SYSTEM_MODE == "historical":
        print("\nExecution Mode:")
        print("Demonstration Using Historical Weather Dataset")

        print("\nData Source:")
        print(
            sample_data.get(
                "data_source",
                "GeoSphere Austria "
                "(Graz Universität Station)",
            )
        )

        print("\nInput Record:")
        if DEMO_MODE == "frost":
            print(
                "Historical Frost Event "
                "(Period: 1990 - 2024)"
            )
        else:
            print(
                "Recent Historical Observation "
                "(Period: 1990 - 2024)"
            )

        print("\nNote:")
        print(
            "This demonstration uses a historical "
            "weather observation"
        )
        print(
            "from the training dataset rather than "
            "a live forecast."
        )

    elif SYSTEM_MODE == "live":
        print("\nExecution Mode:")
        print("Operational Forecast Using Live Weather Data")

        print("\nData Source:")
        print(
            sample_data.get(
                "data_source",
                "Open-Meteo Forecast API",
            )
        )

        print("\nLocation:")
        print(
            sample_data.get(
                "location",
                "Graz, Austria",
            )
        )

        print("\nForecast Generated:")
        print(
            sample_data.get(
                "forecast_generated",
                "Unavailable",
            )
        )

        print("\nPrediction Date:")
        print(
            sample_data.get(
                "prediction_date",
                "Unavailable",
            )
        )

        print("\nForecast Window:")
        start = sample_data.get(
            "forecast_window_start",
            "Unavailable",
        )
        end = sample_data.get(
            "forecast_window_end",
            "Unavailable",
        )
        print(f"{start} to {end}")

        print("\nModel:")
        print("Random Forest Frost Prediction Model")

        print("\nNote:")
        print(
            "This analysis uses live forecast data "
            "retrieved from the Open-Meteo API."
        )

    print("\n" + "=" * 60)


def print_example_questions() -> None:
    """Print example questions for the active execution mode."""

    print("\nExample Questions:")

    if SYSTEM_MODE == "historical":
        print(
            "- Would this historical weather record "
            "result in frost?"
        )
        print(
            "- Assess the frost risk of this "
            "historical observation."
        )
        print(
            "- Should frost protection have been "
            "activated?"
        )
    else:
        print("- Will frost occur tomorrow?")
        print("- What is the soil condition?")
        print(
            "- Should I activate frost protection "
            "tonight?"
        )


def print_agent_selection(
    required_agents: list[str],
) -> None:
    """Print the agents selected by the routing layer."""

    print("\n")
    print("-" * 60)
    print("INTELLIGENT AGENT SELECTION")
    print("-" * 60)

    print("\nAnalyzing user intent...")
    print("Selecting relevant expert agents...\n")

    if not required_agents:
        print(
            "No frost-analysis agents were selected."
        )
        return

    for agent in required_agents:
        print(f"- {agent}")


def print_section(title: str) -> None:
    """Print a section header used before each agent's output block."""

    print("\n")
    print("-" * 60)
    print(title)
    print("-" * 60)


def print_weather_output(
    weather_output: dict[str, Any],
) -> None:
    """Print WeatherAgent output."""

    print_section("WEATHER AGENT")

    probability = weather_output[
        "frost_probability"
    ]
    prediction = weather_output[
        "frost_prediction"
    ]

    print(
        f"\n- Frost Probability: "
        f"{probability * 100:.1f}%"
    )

    print(
        f"- Predicted Frost Event: "
        f"{'YES' if prediction == 1 else 'NO'}"
    )


def print_soil_output(
    soil_output: dict[str, Any],
) -> None:
    """Print SoilAgent output."""

    print_section("SOIL AGENT")

    print(
        f"\n- Soil Temperature: "
        f"{soil_output['soil_temperature']:.1f} °C"
    )

    print(
        f"- Soil Risk: "
        f"{soil_output['soil_risk'].upper()}"
    )


def print_radiation_output(
    radiation_output: dict[str, Any],
) -> None:
    """Print RadiationFrostAgent output."""

    print_section("RADIATION FROST AGENT")

    print(
        f"\n- Radiation Frost Risk: "
        f"{radiation_output['risk_level']}"
    )


def print_planner_output(
    planner_output: dict[str, Any],
) -> None:
    """Print PlannerAgent output."""

    print_section("PLANNER AGENT")

    print(
        f"\n- Priority Level: "
        f"{planner_output['priority']}"
    )

    print(
        f"- Recommended Action:\n"
        f"  {planner_output['recommended_action']}"
    )


def main() -> None:
    """Run one interactive frost-analysis workflow."""

    try:
        sample_data = load_input_data()
    except (
        LiveWeatherDataError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print("\n" + "=" * 60)
        print("SYSTEM STARTUP ERROR")
        print("=" * 60)
        print(str(exc))
        raise SystemExit(1) from exc

    print_system_header(sample_data)
    print_example_questions()

    question = input(
        "\nAsk your question:\n> "
    ).strip()

    if not question:
        print(
            "\nNo question was entered. "
            "Please run the system again."
        )
        return

    result = frost_graph.invoke(
        {
            "question": question,
            "sample_data": sample_data,
        }
    )

    weather_output = result.get(
        "weather_output"
    )
    soil_output = result.get(
        "soil_output"
    )
    radiation_output = result.get(
        "radiation_output"
    )
    planner_output = result.get(
        "planner_output"
    )
    required_agents = result.get(
        "required_agents",
        [],
    )

    print("\n")
    print("-" * 60)
    print("USER QUERY")
    print("-" * 60)
    print(question)

    print_agent_selection(required_agents)

    if weather_output is not None:
        print_weather_output(weather_output)

    if soil_output is not None:
        print_soil_output(soil_output)

    if radiation_output is not None:
        print_radiation_output(
            radiation_output
        )

    if planner_output is not None:
        print_planner_output(
            planner_output
        )

    print("\n")
    print("-" * 60)
    print("AI DECISION SUPPORT ENGINE")
    print("-" * 60)

    if planner_output is not None:
        print(
            "\nGenerating decision support "
            "recommendation..."
        )
    else:
        print(
            "\nGenerating frost risk assessment..."
        )

    print("\n")
    print("=" * 60)
    print("FINAL DECISION SUPPORT REPORT")
    print("=" * 60)

    print(
        result.get(
            "final_response",
            "No final response was generated.",
        )
    )

    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
