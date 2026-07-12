"""Classifier and end-to-end pipeline construction."""

from __future__ import annotations

from pathlib import Path

from joblib import Memory
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_STATE
from .features import SafeSelectKBest, build_feature_union


def build_pipeline(
    model_name: str,
    feature_name: str,
    cache_dir: str | Path,
    quick: bool = False,
) -> Pipeline:
    """Create a leak-safe text features + classifier pipeline."""
    selector_k = 1000 if quick else 4000

    if model_name == "naive_bayes":
        classifier = MultinomialNB()
    elif model_name == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
        selector_k = 800 if quick else 2500
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        selector_k = 800 if quick else 2500
    elif model_name == "neural_network":
        classifier = MLPClassifier(
            random_state=RANDOM_STATE,
            solver="adam",
            batch_size=256 if quick else 128,
            max_iter=10 if quick else 100,
            early_stopping=False,
            n_iter_no_change=8,
        )
        selector_k = 300 if quick else 3000
    else:
        raise ValueError(f"Unknown model: {model_name}")

    memory = Memory(location=str(cache_dir), verbose=0)
    return Pipeline(
        steps=[
            ("features", build_feature_union(feature_name, quick=quick)),
            ("selection", SafeSelectKBest(k=selector_k)),
            ("classifier", classifier),
        ],
        memory=memory,
    )
