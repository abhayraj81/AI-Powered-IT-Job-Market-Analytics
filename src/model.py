"""
Machine Learning component: IT Role Category Classifier.

Predicts the INFERRED category of an IT role from its job description.

IMPORTANT - what this model does and does not show
--------------------------------------------------
Categories are produced by keyword rules on the job TITLE
(preprocessing.infer_category). If descriptions repeat title words, a classifier
can score well just by rediscovering those rules. Two safeguards are built in:

  1. Title words are REMOVED from each description before training (mask_title=True).
  2. title_leakage_check() reports cross-validated scores with and without the
     title words, so the size of the leak is measured instead of assumed.

Evaluation protocol (fixes the old single-split / test-set-selection problem)
  * one stratified 80/20 split, test set touched ONCE;
  * model chosen by stratified k-fold CV macro-F1 on the TRAINING part only;
  * a majority-class baseline is reported next to every model;
  * the model that is reported as best is the model that is saved and served;
  * all reported numbers are stored with the model, so the app can show them
    after a restart.

Run `python -m src.model` from the project root to (re)train and print results.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.nlp import clean_for_nlp

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "it_classifier_pipeline.pkl"
RESULTS_DIR = ROOT / "outputs" / "results"

MODEL_VERSION = 3                   # bump to force a retrain when the logic changes
RANDOM_STATE = 42
MIN_CATEGORY_SAMPLES = 10           # categories with fewer roles are excluded from ML
EXCLUDED_CATEGORIES = {"Other IT"}  # catch-all bucket, not a real class
BASELINE_NAME = "Majority-class baseline"
_SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro"}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _masked_text(description: str, title: str) -> str:
    """Cleaned description with every token that also occurs in the title removed."""
    title_tokens = set(clean_for_nlp(title).split())
    return " ".join(t for t in clean_for_nlp(description).split() if t not in title_tokens)


def prepare_features(df: pd.DataFrame, mask_title: bool = True,
                     min_samples: int = MIN_CATEGORY_SAMPLES):
    """Return (X, y, categories) for the categories that have enough roles."""
    work = df[~df["category"].isin(EXCLUDED_CATEGORIES)]
    work = work[work["job_description"].str.strip() != ""]
    counts = work["category"].value_counts()
    valid = counts[counts >= min_samples].index
    work = work[work["category"].isin(valid)]

    if mask_title:
        X = pd.Series([_masked_text(d, t) for d, t in zip(work["job_description"], work["job_title"])])
    else:
        X = pd.Series([clean_for_nlp(d) for d in work["job_description"]])
    y = work["category"].reset_index(drop=True)
    return X, y, sorted(valid)


def split_data(X: pd.Series, y: pd.Series, test_size: float = 0.20,
               random_state: int = RANDOM_STATE):
    """Single stratified split. Returns X_train, X_test, y_train, y_test."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(splitter.split(X, y))
    return (
        X.iloc[idx_train].reset_index(drop=True),
        X.iloc[idx_test].reset_index(drop=True),
        y.iloc[idx_train].reset_index(drop=True),
        y.iloc[idx_test].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def get_classifiers() -> dict:
    """Fresh classifier instances (no shared state between runs)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", class_weight="balanced",
            random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(
            max_iter=5000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(alpha=0.5),
    }


def build_pipeline(classifier) -> Pipeline:
    """TF-IDF (on already-cleaned text) -> classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, token_pattern=r"\S+")),
        ("clf", classifier),
    ])


def _make_cv(y: pd.Series, n_splits: int = 5) -> StratifiedKFold:
    smallest = int(y.value_counts().min())
    return StratifiedKFold(n_splits=max(2, min(n_splits, smallest)),
                           shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def cross_validate_models(X_train: pd.Series, y_train: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Stratified k-fold CV (training data only) for every model + the baseline."""
    cv = _make_cv(y_train, n_splits)
    rows = []
    for name, clf in get_classifiers().items():
        res = cross_validate(build_pipeline(clf), X_train, y_train, cv=cv, scoring=_SCORING)
        rows.append({
            "Model": name,
            "CV Accuracy": round(float(np.mean(res["test_accuracy"])), 4),
            "CV Accuracy SD": round(float(np.std(res["test_accuracy"])), 4),
            "CV Macro F1": round(float(np.mean(res["test_macro_f1"])), 4),
            "CV Macro F1 SD": round(float(np.std(res["test_macro_f1"])), 4),
        })
    base = cross_validate(DummyClassifier(strategy="most_frequent"),
                          np.zeros((len(y_train), 1)), y_train, cv=cv, scoring=_SCORING)
    rows.append({
        "Model": BASELINE_NAME,
        "CV Accuracy": round(float(np.mean(base["test_accuracy"])), 4),
        "CV Accuracy SD": round(float(np.std(base["test_accuracy"])), 4),
        "CV Macro F1": round(float(np.mean(base["test_macro_f1"])), 4),
        "CV Macro F1 SD": round(float(np.std(base["test_macro_f1"])), 4),
    })
    out = pd.DataFrame(rows)
    out.attrs["n_folds"] = cv.get_n_splits()
    return out


def evaluate_on_test(X_train, X_test, y_train, y_test):
    """Fit every model on the training split and score the held-out test split ONCE."""
    rows, preds = [], {}
    for name, clf in get_classifiers().items():
        pipe = build_pipeline(clf).fit(X_train, y_train)
        preds[name] = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds[name]), 4),
            "Macro F1": round(f1_score(y_test, preds[name], average="macro", zero_division=0), 4),
        })
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    base_pred = dummy.predict(np.zeros((len(y_test), 1)))
    rows.append({
        "Model": BASELINE_NAME,
        "Accuracy": round(accuracy_score(y_test, base_pred), 4),
        "Macro F1": round(f1_score(y_test, base_pred, average="macro", zero_division=0), 4),
    })
    return pd.DataFrame(rows), preds


def title_leakage_check(df: pd.DataFrame, model_name: str = "Logistic Regression",
                        n_splits: int = 5) -> pd.DataFrame:
    """CV scores with title words KEPT vs REMOVED. A big drop means the model leans on the title."""
    rows = []
    for masked in (False, True):
        X, y, _ = prepare_features(df, mask_title=masked)
        cv = _make_cv(y, n_splits)
        res = cross_validate(build_pipeline(get_classifiers()[model_name]), X, y, cv=cv, scoring=_SCORING)
        rows.append({
            "Title words in descriptions": "removed" if masked else "kept",
            "CV Accuracy": round(float(np.mean(res["test_accuracy"])), 4),
            "CV Macro F1": round(float(np.mean(res["test_macro_f1"])), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Train / save / load
# ---------------------------------------------------------------------------

def train_and_save(df: pd.DataFrame, path: Path = MODEL_PATH, mask_title: bool = True) -> dict:
    """
    Full protocol: split -> CV model selection on train -> single test evaluation ->
    refit the SELECTED model on all data -> save {"pipeline", "meta"}.
    """
    X, y, categories = prepare_features(df, mask_title=mask_title)
    X_train, X_test, y_train, y_test = split_data(X, y)

    cv_df = cross_validate_models(X_train, y_train)
    candidates = cv_df[cv_df["Model"] != BASELINE_NAME]
    best_name = candidates.sort_values("CV Macro F1", ascending=False).iloc[0]["Model"]

    test_df, preds = evaluate_on_test(X_train, X_test, y_train, y_test)
    test_df["Selected"] = test_df["Model"] == best_name
    cv_df["Selected"] = cv_df["Model"] == best_name

    report = classification_report(y_test, preds[best_name], labels=categories,
                                   output_dict=True, zero_division=0)

    serving = build_pipeline(get_classifiers()[best_name]).fit(X, y)

    meta = {
        "version": MODEL_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(X)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_classes": len(categories), "classes": categories,
        "mask_title": mask_title,
        "cv_folds": int(cv_df.attrs.get("n_folds", 5)),
        "selected_model": best_name,
        "cv_results": cv_df.to_dict("records"),
        "test_results": test_df.to_dict("records"),
        "y_test": y_test.tolist(), "y_pred": [str(p) for p in preds[best_name]],
        "classification_report": report,
    }
    bundle = {"pipeline": serving, "meta": meta}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def load_bundle(path: Path = MODEL_PATH):
    """Return the saved bundle, or None if missing / outdated / unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    if (isinstance(obj, dict) and "pipeline" in obj
            and obj.get("meta", {}).get("version") == MODEL_VERSION):
        return obj
    return None  # e.g. an old bare-Pipeline pickle -> retrain


def get_or_train_bundle(df: pd.DataFrame, path: Path = MODEL_PATH) -> dict:
    return load_bundle(path) or train_and_save(df, path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_category(pipeline, job_description: str) -> str:
    return pipeline.predict([clean_for_nlp(job_description)])[0]


def predict_proba_category(pipeline, job_description: str) -> pd.DataFrame:
    """
    Per-category confidence, sorted high to low.
    Uses predict_proba when the classifier has it; for LinearSVC it converts the
    decision scores with a softmax (a relative confidence, not a calibrated probability).
    """
    cleaned = clean_for_nlp(job_description)
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        scores = pipeline.predict_proba([cleaned])[0]
    elif hasattr(clf, "decision_function"):
        raw = np.atleast_2d(pipeline.decision_function([cleaned]))[0]
        e = np.exp(raw - raw.max())
        scores = e / e.sum()
    else:
        pred = pipeline.predict([cleaned])[0]
        return pd.DataFrame({"Category": [pred], "Confidence (%)": [100.0]})
    out = pd.DataFrame({"Category": clf.classes_, "Confidence (%)": np.round(scores * 100, 1)})
    return out.sort_values("Confidence (%)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, categories: list):
    cm = confusion_matrix(y_test, y_pred, labels=categories)
    fig = px.imshow(
        cm, x=categories, y=categories, color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Confusion Matrix - held-out test set", text_auto=True, aspect="auto",
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", xaxis_tickangle=-25)
    return fig


def plot_model_comparison(results_df: pd.DataFrame):
    melted = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1"],
                             var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric", barmode="group", text="Score",
        color_discrete_sequence=["#3b82d4", "#7c5cd8"],
        title="Held-out test scores (baseline shown for reference)",
        range_y=[0, 1.05],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# CLI:  python -m src.model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.preprocessing import load_and_clean

    data = load_and_clean(str(ROOT / "data" / "IT_Job_Roles_Skills.csv"))
    result = train_and_save(data)
    m = result["meta"]
    cv_table, test_table = pd.DataFrame(m["cv_results"]), pd.DataFrame(m["test_results"])
    print(f"\nSelected model (by {m['cv_folds']}-fold CV macro-F1): {m['selected_model']}")
    print(f"Roles used: {m['n_samples']} ({m['n_train']} train / {m['n_test']} test), "
          f"{m['n_classes']} classes, title words removed: {m['mask_title']}")
    print("\nCross-validation (training split):\n", cv_table.to_string(index=False))
    print("\nHeld-out test set:\n", test_table.to_string(index=False))
    print("\nTitle-leakage check (5-fold CV, Logistic Regression):\n",
          title_leakage_check(data).to_string(index=False))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(RESULTS_DIR / "model_cv_results.csv", index=False)
    test_table.to_csv(RESULTS_DIR / "model_test_results.csv", index=False)
    print(f"\nSaved model to {MODEL_PATH} and tables to {RESULTS_DIR}")
