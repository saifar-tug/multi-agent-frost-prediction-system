# src/main.py

from agents.weather_agent import WeatherAgent
from agents.soil_agent import SoilAgent
from agents.planner_agent import PlannerAgent
from agents.llm_coordinator import LLMCoordinator
from utils.data_generator import generate_weather_data


def main():
    dataset = generate_weather_data()

    weather_agent = WeatherAgent(dataset)
    soil_agent = SoilAgent()
    planner_agent = PlannerAgent()
    llm = LLMCoordinator()

    temperature = -1
    humidity = 0.8
    soil_temperature = 0.5
    soil_moisture = 0.8

    weather_result = weather_agent.predict_frost(temperature, humidity)
    soil_result = soil_agent.assess_vulnerability(soil_temperature, soil_moisture)
    planner_result = planner_agent.decide_action(
        weather_result["frost_probability"],
        soil_result["vulnerability_score"]
    )

    summary = llm.generate_summary(weather_result, soil_result, planner_result)

    print("\n=== Multi-Agent Frost System ===")
    print(summary)


if __name__ == "__main__":
    main()