# src/main.py

from agents.weather_agent import WeatherAgent
from agents.soil_agent import SoilAgent
from agents.planner_agent import PlannerAgent
from agents.llm_coordinator import LLMCoordinator
from utils.data_loader import load_graz_frost_dataset


def main():
    dataset = load_graz_frost_dataset("data/raw/graz_tmin.txt")

    print("\n=== Multi-Agent Frost System (1-Day Ahead Forecast) ===\n")

    print("Dataset statistics:")
    print("Total samples:", len(dataset))
    print("Date range:", dataset["DATE"].min().date(),
          "to", dataset["DATE"].max().date())
    print("Frost days (target):", dataset["target_frost"].sum())
    print("Non-frost days:", len(dataset) - dataset["target_frost"].sum())
    print()

    weather_agent = WeatherAgent(dataset)
    soil_agent = SoilAgent()
    planner_agent = PlannerAgent()
    llm = LLMCoordinator()

    # Example: simulate today's minimum temperature
    temperature_today = -2.0
    soil_temperature = 2.0
    soil_moisture = 0.6

    weather_result = weather_agent.predict_frost(temperature_today)
    soil_result = soil_agent.assess_vulnerability(
        soil_temperature,
        soil_moisture
    )

    planner_result = planner_agent.decide_action(
        weather_result["frost_probability"],
        soil_result["vulnerability_score"]
    )

    summary = llm.generate_summary(
        weather_result,
        soil_result,
        planner_result
    )

    print("=== System Decision ===")
    print(summary)


if __name__ == "__main__":
    main()