import joblib
import pandas as pd

from src.data_pipeline.live_data_loader import (
    build_feature_row
)


model = joblib.load(
    "models/random_forest_frost_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

live_row = build_feature_row()

X_live = pd.DataFrame(
    [live_row]
)

X_live = X_live[
    feature_columns
]

probability = (
    model.predict_proba(X_live)[0][1]
)

prediction = (
    model.predict(X_live)[0]
)

print("\nLIVE FROST PREDICTION\n")

print(
    f"Probability: {probability:.3f}"
)

print(
    f"Prediction: {prediction}"
)