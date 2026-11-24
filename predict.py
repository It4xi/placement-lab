"""Command line prediction: python predict.py --sample or --input student.json."""

import argparse
import json
from pathlib import Path

import joblib

from src.placement import validate_input

SAMPLE = {
    "cgpa": 8.1, "aptitude_score": 76, "coding_score": 82,
    "communication_score": 72, "internships": 1, "projects": 3,
    "certifications": 2, "backlogs": 0,
    "specialization": "AI/ML", "placement_training": "Yes",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate fictional placement likelihood")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="Use the bundled example")
    group.add_argument("--input", type=Path, help="JSON file with one student's features")
    args = parser.parse_args()
    model_path = Path(__file__).parent / "artifacts" / "placement_model.joblib"
    if not model_path.exists():
        parser.error("Model missing. Run python train.py first.")
    row = SAMPLE if args.sample else json.loads(args.input.read_text(encoding="utf-8"))
    x = validate_input(row)
    model = joblib.load(model_path)
    probability = float(model.predict_proba(x)[0, 1])
    print(json.dumps({"prediction": "Placed" if probability >= 0.5 else "Not placed",
                      "placement_probability": round(probability, 4),
                      "threshold": 0.5}, indent=2))


if __name__ == "__main__":
    main()
