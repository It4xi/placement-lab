# Placement Lab

Placement Lab is an interactive machine learning demo for exploring student placement estimates on synthetic data. Change an input such as coding score to see how the model responds, or adjust the decision threshold to see how false positives and false negatives change.

## Try the demo

The browser demo runs client-side with a pre-exported model. Serve it locally from the repository root and open [localhost:8000](http://127.0.0.1:8000).

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory web_demo/dist
```

Choose a fictional profile, apply it, and move the coding-score slider. The scenario chart varies one input at a time, and you can download the comparison as JSON. The evaluation view includes a confusion matrix, probability calibration, and permutation feature importance.

## Train and run in Python

Use Python 3.12 in a virtual environment.

```bash
python -m pip install -r requirements.txt
python train.py
python predict.py --sample
python -m streamlit run app.py
```

To score your own fictional example, edit `sample_student.json` and run `python predict.py --input sample_student.json`. Run `python export_web_model.py` after training to refresh the browser model. The exporter supports the Logistic Regression pipeline selected by the default training run.

## Model evaluation

The generator creates 12,000 training records and 6,000 independently generated holdout records with ten predictors per profile. Training compares Logistic Regression and Random Forest using mean F1 across five stratified folds. The scikit-learn pipelines fit preprocessing within each fold.

| Model | CV F1 | Holdout F1 | Holdout ROC AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.750 | 0.761 | 0.818 |
| Random Forest | 0.745 | 0.751 | 0.806 |

The selected Logistic Regression model reached **73.65% holdout accuracy** and a **0.174 Brier score**. See [recorded metrics](artifacts/metrics.json) for calibration bands, threshold counts, and a training-size comparison.

## Project structure

- `src/` contains synthetic data generation, feature validation, and evaluation logic.
- `train.py`, `predict.py`, and `app.py` provide training, CLI inference, and the Streamlit interface.
- `web_demo/dist/` contains the static browser demo and exported model.
- `data/` and `artifacts/` contain the generated dataset, model, and evaluation results.

This portfolio edition comes from a collaborative coursework project. All student records are fictional. The model reflects assumptions in the synthetic generator and is intended for demonstration.
