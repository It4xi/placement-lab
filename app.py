"""Interactive placement demo. Run with python -m streamlit run app.py."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.placement import SPECIALIZATIONS, validate_input


ROOT = Path(__file__).parent
OUT = ROOT / "artifacts"
st.set_page_config(page_title="Placement Lab", page_icon="🎓", layout="wide")
st.title("Placement Lab")
st.write("Explore how academic performance, skills and experience affect a simulated placement prediction.")

if not (OUT / "metrics.json").exists() or not (OUT / "placement_model.joblib").exists():
    st.error("Run python train.py to build the model and evaluation data.")
    st.stop()
model = joblib.load(OUT / "placement_model.joblib")
results = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
selected = results["selected_model"]
diagnostics = results["diagnostics"]
profile_tab, evidence_tab = st.tabs(["Student simulator", "Model evaluation"])

# (column name, display label, minimum, maximum, default, increment)
FIELDS = [
    ("cgpa", "CGPA", 4.0, 10.0, 7.2, 0.1),
    ("coding_score", "Coding score", 0, 100, 60, 1),
    ("aptitude_score", "Aptitude score", 0, 100, 60, 1),
    ("communication_score", "Communication score", 0, 100, 65, 1),
    ("internships", "Internships", 0, 3, 1, 1),
    ("projects", "Projects", 0, 6, 2, 1),
    ("certifications", "Certifications", 0, 5, 1, 1),
    ("backlogs", "Backlogs", 0, 5, 1, 1),
]

with profile_tab:
    inputs, simulator = st.columns([1, 1.25], gap="large")
    with inputs:
        st.subheader("Student profile")
        with st.form("profile"):
            row = {}
            columns = st.columns(2)
            for index, (key, label, low, high, default, step) in enumerate(FIELDS):
                row[key] = columns[index % 2].number_input(label, low, high, default, step)
            row["specialization"] = st.selectbox("Specialization", SPECIALIZATIONS)
            row["placement_training"] = st.radio("Placement training", ["Yes", "No"], horizontal=True)
            applied = st.form_submit_button("Apply profile", type="primary")
        if applied or "active_profile" not in st.session_state:
            st.session_state.active_profile = row.copy()

    profile = st.session_state.active_profile
    current = validate_input(profile)
    base = float(model.predict_proba(current)[0, 1])
    with simulator:
        st.subheader("What if one input changed?")
        feature_label = st.selectbox("Input to explore", [f[1] for f in FIELDS], index=1)
        key, _, low, high, _, step = next(f for f in FIELDS if f[1] == feature_label)
        value = st.slider("Scenario value", low, high, profile[key], step, key=f"scenario_{key}_{profile[key]}")
        scenario = {**profile, key: value}
        score = float(model.predict_proba(validate_input(scenario))[0, 1])
        a, b = st.columns(2)
        a.metric("Applied profile", f"{base:.1%}", border=True)
        b.metric("Scenario estimate", f"{score:.1%}", f"{100*(score-base):+.1f} pp", border=True)
        values = np.linspace(low, high, round((high-low)/step)+1).round(2)
        sweep = pd.concat([current] * len(values), ignore_index=True)
        sweep[key] = values
        line = pd.DataFrame({feature_label: values, "Placement estimate (%)": model.predict_proba(sweep)[:, 1]*100})
        st.line_chart(line, x=feature_label, y="Placement estimate (%)", color="#08766F", height=190)
        st.caption(f"{feature_label} changes from {profile[key]:g} to {value:g}. All other applied values stay fixed. pp = percentage points.")
        verdict = "Placed" if score >= 0.5 else "Not placed"
        st.write(f"Scenario classification at the 50% cutoff: **{verdict}**")
        st.download_button("Save comparison", json.dumps({
            "model": selected, "data_source": "synthetic", "cutoff": 0.5,
            "applied": {"features": profile, "probability": base},
            "scenario": {"features": scenario, "probability": score, "classification": verdict}
        }, indent=2), "scenario.json", "application/json")
    st.caption("Each estimate describes the model’s response to a fictional profile. Real placement outcomes require validation on real cohorts.")

with evidence_tab:
    test = results["models"][selected]["test"]
    st.subheader("How well does the model predict?")
    st.caption(f"{selected} · Selected by five-fold training F1 · {results['test_rows']} held-out fictional students")
    left, middle, right = st.columns(3)
    left.metric("Accuracy", f"{test['accuracy']:.1%}",
                f"{100*(test['accuracy']-diagnostics['baseline_accuracy']):+.1f} pp vs majority", border=True)
    middle.metric("F1 at 50%", f"{test['f1']:.3f}", border=True)
    right.metric("Brier score", f"{diagnostics['brier']:.3f}", border=True)
    cutoff_col, calib_col = st.columns(2, gap="large")
    with cutoff_col:
        st.subheader("Decision cutoff")
        cutoff = st.slider("Classify as placed at or above (%)", 10, 95, 50, 5)
        counts = next(t for t in diagnostics["thresholds"] if t["cutoff"] == cutoff)
        tn, fp, fn, tp = [counts[k] for k in ("tn", "fp", "fn", "tp")]
        matrix = pd.DataFrame([[tn, fp], [fn, tp]],
            index=["Actual: not placed", "Actual: placed"], columns=["Predict: not placed", "Predict: placed"])
        st.table(matrix)
        precision = f"{tp/(tp+fp):.1%}" if tp + fp else "undefined"
        st.caption(f"Precision {precision} · Recall {tp/(tp+fn):.1%}. This control explores the test cohort. The student simulator uses 50%.")
    with calib_col:
        st.subheader("Do the probabilities match outcomes?")
        bands = pd.DataFrame(diagnostics["calibration"]).set_index("band")
        rates = bands[["mean_score", "observed_rate"]].mul(100)
        rates.columns = ["Mean estimate (%)", "Observed placed (%)"]
        st.bar_chart(rates, stack=False, height=200, color=["#08766F", "#CA7139"])
        st.caption(f"Students are grouped by predicted probability. Lower Brier scores indicate smaller probability errors. A constant training-rate forecast scores {diagnostics['baseline_brier']:.3f}.")
    st.subheader("Which features influence the model?")
    st.image(str(OUT / "feature_importance.png"), width=640)
    st.caption("Each bar shows the average change in test F1 after shuffling one input across students. A larger drop indicates greater model reliance.")
    with st.expander("Model comparison and calibration counts"):
        comparison = []
        for name, record in results["models"].items():
            comparison.append({"Model": name, "Training CV F1": record["cv_f1_mean"],
                **{k: record["test"][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
        st.dataframe(pd.DataFrame(comparison).set_index("Model").round(3))
        st.dataframe(bands.round(3))
    with st.expander("What changed when we trained on more data?"):
        scale = results["training_size_comparison"]
        rows = [{"Training profiles": scale[f"{version}_train_rows"],
                 "Accuracy": f"{scale[f'{version}_accuracy']:.2%}",
                 "Brier score": f"{scale[f'{version}_brier']:.4f}"}
                for version in ("earlier", "expanded")]
        st.caption("Both Logistic Regression models were evaluated on the same holdout. Lower Brier scores are better.")
        st.table(pd.DataFrame(rows).set_index("Training profiles"))
        low, high = scale["brier_reduction_ci95"]
        st.caption(f"Paired 95% interval for the Brier score reduction is {low:.4f} to {high:.4f}.")

