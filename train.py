"""Generate fictional data, compare models, evaluate, and save artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.placement import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                           SEED, TARGET, generate_dataset)
from src.diagnostics import probability_diagnostics, training_size_comparison

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts"
TRAIN_ROWS = 12_000
TEST_ROWS = 6_000
TRAIN_SEED = SEED + 1
TEST_SEED = SEED + 3


def make_pipeline(estimator):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    category = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                                    ("categorical", category, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


def evaluate(model, x, y):
    prediction = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return {"accuracy": accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def main():
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    train = generate_dataset(TRAIN_ROWS, TRAIN_SEED)
    x_train, y_train = train[FEATURES], train[TARGET]

    models = {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
        "Random Forest": make_pipeline(RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=-1)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # Select only from training data. The holdout is touched after selection.
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_results[name] = {"cv_f1_mean": float(scores.mean()),
                            "cv_f1_std": float(scores.std())}
    selected = max(cv_results, key=lambda name: cv_results[name]["cv_f1_mean"])

    # The larger training cohort uses its own generator seed. No model or
    # training size is selected using this separate final cohort.
    test = generate_dataset(TEST_ROWS, TEST_SEED)
    x_test, y_test = test[FEATURES], test[TARGET]
    data = pd.concat([train, test], ignore_index=True)
    data.to_csv(DATA / "synthetic_placement.csv", index=False)

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = {**cv_results[name], "test": evaluate(model, x_test, y_test)}

    best = models[selected]
    previous_data = generate_dataset()
    previous_x, _, previous_y, _ = train_test_split(
        previous_data[FEATURES], previous_data[TARGET],
        test_size=0.20, random_state=SEED, stratify=previous_data[TARGET])
    logistic = models["Logistic Regression"]
    previous = clone(logistic).fit(previous_x, previous_y)
    scale_comparison = training_size_comparison(
        y_test, previous.predict_proba(x_test)[:, 1],
        logistic.predict_proba(x_test)[:, 1], seed=TEST_SEED)
    scale_comparison["earlier_train_rows"] = len(previous_x)
    scale_comparison["expanded_train_rows"] = len(x_train)

    joblib.dump(best, OUT / "placement_model.joblib")
    imp = permutation_importance(best, x_test, y_test, n_repeats=12,
                                 random_state=SEED, scoring="f1", n_jobs=1)
    importance = pd.DataFrame({"feature": FEATURES,
                               "mean_f1_drop": imp.importances_mean,
                               "std_f1_drop": imp.importances_std})
    importance = importance.sort_values("mean_f1_drop", ascending=False)
    importance.to_csv(OUT / "permutation_importance.csv", index=False)

    metrics = {"dataset_type": "synthetic", "seed": SEED,
               "training_seed": TRAIN_SEED, "holdout_seed": TEST_SEED, "total_rows": len(data),
               "train_rows": len(x_train), "test_rows": len(x_test),
               "positive_rate": float(data[TARGET].mean()), "selected_model": selected,
               "selection_rule": "highest mean 5-fold training CV F1",
               "diagnostics": probability_diagnostics(
                   y_train, y_test, best.predict_proba(x_test)[:, 1]),
               "training_size_comparison": scale_comparison,
               "models": results}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    ordered = importance.iloc[::-1]
    ax.barh(ordered["feature"], ordered["mean_f1_drop"], color="#2268A8")
    ax.set_xlabel("Mean decrease in holdout F1 when shuffled")
    ax.set_title("Permutation feature importance")
    ax.axvline(0, color="black", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.7, 4.4))
    ConfusionMatrixDisplay.from_predictions(y_test, best.predict(x_test),
        display_labels=["Not placed", "Placed"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Holdout confusion matrix: {selected}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
