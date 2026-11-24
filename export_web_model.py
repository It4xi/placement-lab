"""Export the fitted placement model for the static browser demo."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.placement import CATEGORICAL_FEATURES, NUMERIC_FEATURES, mid_prep_profile, validate_input

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def main():
    pipeline = joblib.load(ARTIFACTS / "placement_model.joblib")
    transform, classifier = pipeline.named_steps.values()
    if not isinstance(classifier, LogisticRegression):
        raise ValueError("The browser export supports the selected Logistic Regression model.")
    scale = transform.named_transformers_["numeric"].named_steps["scale"]
    encode = transform.named_transformers_["categorical"].named_steps["encode"]
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "sample_student.json").read_text(encoding="utf-8"))
    payload = {
        "model": metrics["selected_model"], "testRows": metrics["test_rows"],
        "metrics": metrics["models"][metrics["selected_model"]]["test"],
        "diagnostics": metrics["diagnostics"],
        "trainingComparison": metrics["training_size_comparison"],
        "importance": pd.read_csv(ARTIFACTS / "permutation_importance.csv").to_dict("records"),
        "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
        "means": scale.mean_.tolist(), "scales": scale.scale_.tolist(),
        "categories": [a.tolist() for a in encode.categories_],
        "weights": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
        "sample": profile, "midProfile": mid_prep_profile(profile),
        "sampleProbability": float(pipeline.predict_proba(validate_input(profile))[0, 1]),
    }
    path = ROOT / "web_demo" / "dist" / "model.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
