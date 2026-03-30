import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class WeatherAgent:
    def __init__(self, dataset: pd.DataFrame):
        self.model = LogisticRegression()
        self.train_model(dataset)

    def train_model(self, dataset: pd.DataFrame):
        X = dataset[["TN_C"]]
        y = dataset["target_frost"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"[WeatherAgent] 1-Day Ahead Frost Accuracy: {acc:.4f}")

    def predict_frost(self, temperature_today: float):
        input_df = pd.DataFrame({"TN_C": [temperature_today]})
        frost_prob = self.model.predict_proba(input_df)[0][1]
        frost_prediction = self.model.predict(input_df)[0]

        return {
            "frost_probability": float(frost_prob),
            "frost_prediction": bool(frost_prediction)
        }