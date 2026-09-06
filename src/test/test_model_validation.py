# src/test/test_model_validation.py

from pathlib import Path

import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.data_pipeline.preprocess_data import (
    FEATURE_COLUMNS,
    RENAME_MAP,
)


DATA_PATH = Path(
    "data/raw/"
    "Messstationen Tagesdaten v2 Datensatz_20000101_20260422.csv"
)

N_SPLITS = 5


def load_validation_data():

    df = pd.read_csv(DATA_PATH)

    # Convert the timestamp so chronological ordering is explicit.
    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    df = df.sort_values("time").reset_index(
        drop=True
    )

    # Use the same column mapping as the training pipeline.
    df = df.rename(
        columns=RENAME_MAP
    )

    # Frost is the target and therefore must be available.
    df = df.dropna(
        subset=["frost"]
    ).reset_index(drop=True)

    X = df[FEATURE_COLUMNS].copy()

    y = df["frost"].astype(int)

    dates = df["time"].copy()

    return X, y, dates


def create_pipeline(
    model_name,
):

    if model_name == "Logistic Regression":

        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "Random Forest":

        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def evaluate_model_mean_metrics(model_name, X, y, n_splits=N_SPLITS):
    """Same walk-forward validation as main()'s reporting loop, without the printing,
    so both the CLI report and the regression test below run identical folds."""

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_index, test_index in splitter.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline = create_pipeline(model_name)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        fold_metrics.append({
            "auc": roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else float("nan"),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        })

    mean_metrics = pd.DataFrame(fold_metrics).mean()
    return mean_metrics.to_dict()


@pytest.fixture(scope="module")
def mean_metrics_by_model():
    X, y, _dates = load_validation_data()
    return {
        model_name: evaluate_model_mean_metrics(model_name, X, y)
        for model_name in ("Logistic Regression", "Random Forest")
    }


def test_random_forest_meets_baseline_performance(mean_metrics_by_model):
    metrics = mean_metrics_by_model["Random Forest"]
    assert metrics["auc"] > 0.90
    assert metrics["precision"] > 0.55
    assert metrics["f1"] > 0.65


def test_random_forest_beats_logistic_regression_on_f1(mean_metrics_by_model):
    """Random Forest is the operational model because of its stronger precision/F1 balance."""
    assert mean_metrics_by_model["Random Forest"]["f1"] > mean_metrics_by_model["Logistic Regression"]["f1"]


def test_logistic_regression_has_higher_recall(mean_metrics_by_model):
    """The documented tradeoff: Logistic Regression misses fewer frost events at the cost of more false alarms."""
    assert mean_metrics_by_model["Logistic Regression"]["recall"] > mean_metrics_by_model["Random Forest"]["recall"]


def main():

    print("\n" + "=" * 70)
    print("MODEL COMPARISON - TIME-SERIES VALIDATION")
    print("=" * 70)

    X, y, dates = load_validation_data()

    print("\nDataset")
    print(f"Samples: {len(X)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(
        f"Date range: "
        f"{dates.iloc[0].date()} "
        f"to {dates.iloc[-1].date()}"
    )

    print("\nTarget Distribution")
    print(
        f"No Frost: {(y == 0).sum()}"
    )
    print(
        f"Frost: {(y == 1).sum()}"
    )
    print(
        f"Frost Rate: {y.mean() * 100:.2f}%"
    )

    model_names = [
        "Logistic Regression",
        "Random Forest",
    ]

    all_results = []

    for model_name in model_names:

        print("\n" + "=" * 70)
        print(model_name.upper())
        print("=" * 70)

        splitter = TimeSeriesSplit(
            n_splits=N_SPLITS
        )

        model_results = []

        for fold, (
            train_index,
            test_index,
        ) in enumerate(
            splitter.split(X),
            start=1,
        ):

            X_train = X.iloc[
                train_index
            ]
            X_test = X.iloc[
                test_index
            ]

            y_train = y.iloc[
                train_index
            ]
            y_test = y.iloc[
                test_index
            ]

            train_dates = dates.iloc[
                train_index
            ]
            test_dates = dates.iloc[
                test_index
            ]

            pipeline = create_pipeline(
                model_name
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(
                X_test
            )

            probabilities = (
                pipeline.predict_proba(
                    X_test
                )[:, 1]
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0,
            )

            if y_test.nunique() == 2:

                auc = roc_auc_score(
                    y_test,
                    probabilities,
                )

            else:

                auc = float("nan")

            cm = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            )

            tn, fp, fn, tp = cm.ravel()

            fold_result = {
                "model": model_name,
                "fold": fold,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }

            model_results.append(
                fold_result
            )

            all_results.append(
                fold_result
            )

            print("\n" + "-" * 70)
            print(f"FOLD {fold}")
            print("-" * 70)

            print("\nTemporal Split")

            print(
                "Train: "
                f"{train_dates.iloc[0].date()} "
                "to "
                f"{train_dates.iloc[-1].date()}"
            )

            print(
                "Test:  "
                f"{test_dates.iloc[0].date()} "
                "to "
                f"{test_dates.iloc[-1].date()}"
            )

            print("\nSamples")

            print(
                f"Training Samples: "
                f"{len(X_train)}"
            )

            print(
                f"Test Samples: "
                f"{len(X_test)}"
            )

            print(
                f"Test Frost Events: "
                f"{int(y_test.sum())}"
            )

            print(
                f"Test Non-Frost Events: "
                f"{int((y_test == 0).sum())}"
            )

            print("\nPerformance")

            if pd.isna(auc):

                print(
                    "ROC-AUC: "
                    "Not available"
                )

            else:

                print(
                    f"ROC-AUC: {auc:.4f}"
                )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall:    {recall:.4f}"
            )

            print(
                f"F1 Score:  {f1:.4f}"
            )

            print("\nConfusion Matrix")

            print(
                f"True Negatives:  {tn}"
            )

            print(
                f"False Positives: {fp}"
            )

            print(
                f"False Negatives: {fn}"
            )

            print(
                f"True Positives:  {tp}"
            )

        model_df = pd.DataFrame(
            model_results
        )

        print("\n" + "-" * 70)
        print(f"{model_name.upper()} SUMMARY")
        print("-" * 70)

        print(
            f"Mean ROC-AUC: "
            f"{model_df['auc'].mean():.4f}"
        )

        print(
            f"Mean Precision: "
            f"{model_df['precision'].mean():.4f}"
        )

        print(
            f"Mean Recall: "
            f"{model_df['recall'].mean():.4f}"
        )

        print(
            f"Mean F1 Score: "
            f"{model_df['f1'].mean():.4f}"
        )

    results_df = pd.DataFrame(
        all_results
    )

    comparison_df = (
        results_df
        .groupby("model", sort=False)
        [
            [
                "auc",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "auc": "ROC-AUC",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
            }
        )
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
