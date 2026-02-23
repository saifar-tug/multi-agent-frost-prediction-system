# src/agents/soil_agent.py

class SoilAgent:
    def __init__(self):
        pass

    def assess_vulnerability(self, soil_temperature, soil_moisture):
        """
        Assess crop vulnerability based on soil conditions.
        """
        if soil_temperature <= 1 and soil_moisture > 0.7:
            vulnerability = 0.8
        else:
            vulnerability = 0.3

        return {
            "vulnerability_score": vulnerability
        }