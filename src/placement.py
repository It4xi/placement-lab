"""Synthetic data and a shared input schema for the placement demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 27609
NUMERIC_FEATURES = [
    "cgpa", "aptitude_score", "coding_score", "communication_score",
    "internships", "projects", "certifications", "backlogs",
]
CATEGORICAL_FEATURES = ["specialization", "placement_training"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "placed"
SPECIALIZATIONS = ["AI/ML", "Data Science", "Cybersecurity", "Software Engineering"]


def mid_prep_profile(sample: dict) -> dict:
    """Return the fictional profile used for the report's coding scenario."""
    return {**sample, "cgpa": 7.2, "aptitude_score": 60, "coding_score": 60,
            "communication_score": 65, "internships": 1, "projects": 2,
            "certifications": 1, "backlogs": 1}


def generate_dataset(n: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """Create fictional students; labels are sampled from a documented latent rule.

    The outcome is stochastic, so a model cannot simply memorize a deterministic
    threshold. No real student records or protected personal attributes are used.
    """
    rng = np.random.default_rng(seed)
    cgpa = np.round(np.clip(rng.normal(7.25, 1.05, n), 4.0, 10.0), 2)
    aptitude = np.rint(np.clip(rng.normal(62 + 5 * (cgpa - 7), 15, n), 0, 100)).astype(int)
    coding = np.rint(np.clip(rng.normal(58 + 6 * (cgpa - 7), 18, n), 0, 100)).astype(int)
    communication = np.rint(np.clip(rng.normal(64, 15, n), 0, 100)).astype(int)
    internships = np.clip(rng.poisson(0.75, n), 0, 3)
    projects = np.clip(rng.poisson(2.4, n), 0, 6)
    certifications = np.clip(rng.poisson(1.35, n), 0, 5)
    backlogs = np.clip(rng.poisson(np.maximum(0.12, 1.1 - 0.12 * (cgpa - 6)), n), 0, 5)
    specialization = rng.choice(SPECIALIZATIONS, n)
    training = rng.choice(["Yes", "No"], n, p=[0.63, 0.37])

    # Fictional mechanism: academic, assessment and experience signals affect
    # placement odds. The unobserved noise stands for omitted real-world factors.
    logit = (
        -0.40 + 0.75 * (cgpa - 7.0) + 0.025 * (aptitude - 60)
        + 0.031 * (coding - 60) + 0.020 * (communication - 60)
        + 0.42 * internships + 0.19 * (projects - 2)
        + 0.10 * certifications - 0.47 * backlogs
        + 0.29 * (training == "Yes")
        + rng.normal(0, 0.52, n)
    )
    probability = 1 / (1 + np.exp(-logit))
    placed = rng.binomial(1, probability)
    return pd.DataFrame({
        "cgpa": cgpa, "aptitude_score": aptitude, "coding_score": coding,
        "communication_score": communication, "internships": internships,
        "projects": projects, "certifications": certifications,
        "backlogs": backlogs, "specialization": specialization,
        "placement_training": training, TARGET: placed,
    })


def validate_input(row: dict) -> pd.DataFrame:
    missing = set(FEATURES) - set(row)
    if missing:
        raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
    bounds = {
        "cgpa": (4, 10), "aptitude_score": (0, 100),
        "coding_score": (0, 100), "communication_score": (0, 100),
        "internships": (0, 3), "projects": (0, 6),
        "certifications": (0, 5), "backlogs": (0, 5),
    }
    clean = {}
    for key, (low, high) in bounds.items():
        value = float(row[key])
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        if key != "cgpa" and not value.is_integer():
            raise ValueError(f"{key} must be a whole number")
        clean[key] = value
    if row["specialization"] not in SPECIALIZATIONS:
        raise ValueError(f"specialization must be one of {SPECIALIZATIONS}")
    if row["placement_training"] not in ("Yes", "No"):
        raise ValueError("placement_training must be Yes or No")
    clean["specialization"] = row["specialization"]
    clean["placement_training"] = row["placement_training"]
    return pd.DataFrame([clean], columns=FEATURES)
