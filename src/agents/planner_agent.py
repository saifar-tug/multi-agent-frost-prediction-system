# src/agents/planner_agent.py

class PlannerAgent:
    def __init__(self):
        pass

    def decide_action(self, frost_probability, vulnerability_score):
        """
        Decide contingency action.
        """
        if frost_probability > 0.7 and vulnerability_score > 0.6:
            action = "Activate heating system"
        elif frost_probability > 0.5:
            action = "Send warning alert"
        else:
            action = "No action required"

        return {
            "recommended_action": action
        }