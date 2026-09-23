"""
Machine Learning component: IT Role Category Classifier.

Predicts the inferred category of an IT role from its job description text.
Categories are derived by preprocessing.infer_category() — they are
transparent and deterministic, so the ML task is to learn those patterns.

Models compared:
    - Logistic Regression
    - Linear SVM
    - Multinomial Naive Bayes

Pipeline: TF-IDF vectorisation → Classifier
"""

import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.nlp import clean_for_nlp

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "it_classifier_pipeline.pkl"

MIN_CATEGORY_SAMPLES = 5   # categories with fewer samples are excluded from ML


def prepare_features(df: pd.DataFrame):
    
    cat_counts = df["category"].value_counts()
    valid_cats = cat_counts[cat_counts >= MIN_CATEGORY_SAMPLES].index.tolist()

    sub = df[df["category"].isin(valid_cats)].copy()
    sub = sub[sub["job_description"].str.strip() != ""]

    X = sub["job_description"].apply(clean_for_nlp)
    y = sub["category"]
    return X, y, valid_cats


def split_data(X: pd.Series, y: pd.Series,
               test_size: float = 0.20, random_state: int = 42):
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=random_state)
    idx_train, idx_test = next(splitter.split(X, y))
    return (
        X.iloc[idx_train].reset_index(drop=True),
        X.iloc[idx_test].reset_index(drop=True),
        y.iloc[idx_train].reset_index(drop=True),
        y.iloc[idx_test].reset_index(drop=True),
    )


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42
    ),
    "Linear SVM": LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.5),
}


def _make_pipeline(classifier) -> Pipeline:
    
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )),
        ("clf", classifier),
    ])


def train_and_evaluate_all(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    
    results = []
    for name, clf in CLASSIFIERS.items():
        pipe = _make_pipeline(clf)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results.append({"Model": name,
                         "Accuracy": round(acc, 4),
                         "Macro F1": round(f1, 4)})
        print(f"  [{name}]  Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
    return pd.DataFrame(results)


def train_best_model(X_train, X_test, y_train, y_test,
                     best_name: str = "Logistic Regression"):
    
    clf = CLASSIFIERS[best_name]
    pipe = _make_pipeline(clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    categories = sorted(y_train.unique())
    report = classification_report(y_test, y_pred,
                                    target_names=categories, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=categories))
    return pipe, y_pred, report, categories


def plot_confusion_matrix(y_test, y_pred, categories: list):
    
    cm = confusion_matrix(y_test, y_pred, labels=categories)
    fig = px.imshow(
        cm, x=categories, y=categories,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Confusion Matrix — IT Role Category Classifier",
        text_auto=True, aspect="auto",
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual",
                      xaxis_tickangle=-25)
    return fig


def plot_model_comparison(results_df: pd.DataFrame):
    
    melted = results_df.melt(id_vars="Model",
                              value_vars=["Accuracy", "Macro F1"],
                              var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric",
        barmode="group", text="Score",
        color_discrete_sequence=["#3b82d4", "#7c5cd8"],
        title="Model Comparison — IT Role Classifier",
        range_y=[0, 1.05],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return fig


def save_model(pipeline):
    
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_category(pipeline, job_description: str) -> str:
    
    cleaned = clean_for_nlp(job_description)
    return pipeline.predict([cleaned])[0]


def predict_proba_category(pipeline, job_description: str) -> pd.DataFrame:
    
    cleaned = clean_for_nlp(job_description)
    clf_step = pipeline.named_steps["clf"]
    if hasattr(clf_step, "predict_proba"):
        probs = pipeline.predict_proba([cleaned])[0]
        classes = clf_step.classes_
        df_proba = pd.DataFrame({"Category": classes, "Probability": probs})
        return df_proba.sort_values("Probability", ascending=False).reset_index(drop=True)
    pred = predict_category(pipeline, job_description)
    return pd.DataFrame({"Category": [pred], "Probability": [1.0]})
