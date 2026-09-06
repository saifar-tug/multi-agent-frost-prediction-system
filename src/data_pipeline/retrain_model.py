# src/data_pipeline/retrain_model.py

from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier

from src.data_pipeline.preprocess_data import (
    load_and_preprocess,
    FEATURE_COLUMNS,
)


DATA_PATH = (
    "data/raw/"
    "Messstationen Tagesdaten v2 Datensatz_20000101_20260422.csv"
)

MODEL_DIR = Path("models")


def main():

    MODEL_DIR.mkdir(
        exist_ok=True
    )

    print("=" * 60)
    print("FROST MODEL RETRAINING PIPELINE")
    print("=" * 60)

    print("\nLoading and preprocessing data...")

    X, y = load_and_preprocess(
        DATA_PATH
    )

    print(f"Samples: {len(X)}")
    print(f"Features: {len(X.columns)}")

    print("\nTraining Random Forest model...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X, y)

    print("\nSaving model artifacts...")

    artifacts = {
        MODEL_DIR / "random_forest_frost_model.pkl": rf_model,
        MODEL_DIR / "feature_columns.pkl": FEATURE_COLUMNS,
        MODEL_DIR / "sample_row_recent.pkl": X.iloc[-1].to_dict(),
        MODEL_DIR / "sample_row_frost.pkl": X[y == 1].iloc[0].to_dict(),
    }

    for path, artifact in artifacts.items():
        joblib.dump(artifact, path)

    print("\nSaved:")
    for path in artifacts:
        print(path)

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()