# src/main.py

import joblib

from graph.frost_graph import frost_graph


SYSTEM_MODE = "historical"
# SYSTEM_MODE = "live"

DEMO_MODE = "frost"
# DEMO_MODE = "recent"


def load_demo_sample():

    if DEMO_MODE == "frost":

        return joblib.load(
            "models/sample_row_frost.pkl"
        )

    elif DEMO_MODE == "recent":

        return joblib.load(
            "models/sample_row_recent.pkl"
        )

    raise ValueError(
        f"Unknown DEMO_MODE: {DEMO_MODE}"
    )


def print_system_header():

    print("\n")
    print("=" * 60)
    print("MULTI-AGENT FROST PREDICTION SYSTEM")
    print("=" * 60)

    if SYSTEM_MODE == "historical":

        print("\nExecution Mode:")
        print("Demonstration Using Historical Weather Dataset")

        print("\nData Source:")
        print("GeoSphere Austria (Graz Universität Station)")

        print("\nInput Record:")
        print("Historical Frost Event (Period: 1990 - 2024)")

        print("\nNote:")
        print("This demonstration uses a historical weather observation")
        print("from the training dataset rather than a live weather forecast.")

    else:

        print("\nExecution Mode:")
        print("Operational Forecast")

        print("\nData Source:")
        print("Open-Meteo Forecast API")

        print("\nForecast Date:")
        print("Retrieved dynamically")

        print("\nNote:")
        print("This analysis uses live forecast data.")

    print("\n" + "=" * 60)


def print_example_questions():

    print("\nExample Questions:")

    if SYSTEM_MODE == "historical":

        print("• Would this historical weather record result in frost?")
        print("• Assess the frost risk of this historical observation.")
        print("• Should frost protection have been activated?")

    else:

        print("• Will frost occur tomorrow?")
        print("• What is the soil condition?")
        print("• Should I activate frost protection tonight?")


def main():

    sample_data = load_demo_sample()

    print_system_header()

    print_example_questions()

    question = input(
        "\nAsk your question:\n> "
    )

    result = frost_graph.invoke(
        {
            "question": question,
            "sample_data": sample_data
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

    print("\n")
    print("-" * 60)
    print("USER QUERY")
    print("-" * 60)
    print(question)

    print("\n")
    print("-" * 60)
    print("INTELLIGENT AGENT SELECTION")
    print("-" * 60)

    print("\nAnalyzing user intent...")
    print("Selecting relevant expert agents...\n")

    for agent in result["required_agents"]:

        print(f"✓ {agent}")

    if weather_output is not None:

        print("\n")
        print("-" * 60)
        print("WEATHER AGENT")
        print("-" * 60)

        print(
            f"\n✓ Frost Probability: "
            f"{weather_output['frost_probability'] * 100:.1f}%"
        )

        print(
            f"✓ Predicted Frost Event: "
            f"{'YES' if weather_output['frost_prediction'] == 1 else 'NO'}"
        )

    if soil_output is not None:

        print("\n")
        print("-" * 60)
        print("SOIL AGENT")
        print("-" * 60)

        print(
            f"\n✓ Soil Temperature: "
            f"{soil_output['soil_temperature']:.1f} °C"
        )

        print(
            f"✓ Soil Risk: "
            f"{soil_output['soil_risk'].upper()}"
        )

    if radiation_output is not None:

        print("\n")
        print("-" * 60)
        print("RADIATION FROST AGENT")
        print("-" * 60)

        print(
            f"\n✓ Radiation Frost Risk: "
            f"{radiation_output['risk_level']}"
        )

    if planner_output is not None:

        print("\n")
        print("-" * 60)
        print("PLANNER AGENT")
        print("-" * 60)

        print(
            f"\n✓ Priority Level: "
            f"{planner_output['priority']}"
        )

        print(
            f"✓ Recommended Action:\n"
            f"  {planner_output['recommended_action']}"
        )

    print("\n")
    print("-" * 60)
    print("AI DECISION SUPPORT ENGINE")
    print("-" * 60)

    if planner_output is not None:

        print("\nGenerating decision support recommendation...")

    else:

        print("\nGenerating frost risk assessment...")

    print("\n")
    print("=" * 60)
    print("FINAL DECISION SUPPORT REPORT")
    print("=" * 60)

    print(
        result["final_response"]
    )

    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()