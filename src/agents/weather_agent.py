# src/agents/weather_agent.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class WeatherAgent:
    def __init__(self, dataset):
        self.model = LogisticRegression()
        self.train_model(dataset)

    def train_model(self, dataset):
        X = dataset[["temperature", "humidity"]]
        y = dataset["frost"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"[WeatherAgent] Frost Model Accuracy: {acc:.2f}")

    def predict_frost(self, temperature, humidity):
        X = np.array([[temperature, humidity]])
        frost_prob = self.model.predict_proba(X)[0][1]
        frost_prediction = self.model.predict(X)[0]

        return {
            "frost_probability": float(frost_prob),
            "frost_prediction": bool(frost_prediction)
        }