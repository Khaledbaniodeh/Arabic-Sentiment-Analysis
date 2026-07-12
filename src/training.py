"""Feature comparison, validation tuning, final training, and testing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODELS_DIR,
    FEATURE_CONFIGS,
    MODEL_GRIDS,
    QUICK_MODEL_GRIDS,
    RANDOM_STATE,
)
from .data_loader import random_oversample
from .evaluation import save_model_comparison, save_model_evaluation
from .models import build_pipeline


def _training_view(
    frame: pd.DataFrame,
    balance: bool,
) -> tuple[pd.Series, pd.Series]:
    texts = frame["text"]
    labels = frame["label"]
    if balance:
        return random_oversample(texts, labels, random_state=RANDOM_STATE)
    return texts, labels


def select_feature_set(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    results_dir: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    balance: bool = True,
    quick: bool = False,
) -> str:
    """Compare feature representations using a fixed Naive Bayes baseline."""
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    X_train, y_train = _training_view(train, balance)
    records: list[dict[str, object]] = []

    candidates = ["word", "char", "combined"]
    for feature_name in candidates:
        pipeline = build_pipeline(
            "naive_bayes",
            feature_name,
            cache_dir=Path(cache_dir) / f"feature_{feature_name}",
            quick=quick,
        )
        pipeline.set_params(classifier__alpha=0.5)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(validation["text"])
        elapsed = time.perf_counter() - started
        records.append(
            {
                "feature_set": feature_name,
                "description": FEATURE_CONFIGS[feature_name]["description"],
                "validation_accuracy": float(accuracy_score(validation["label"], predictions)),
                "validation_macro_f1": float(f1_score(validation["label"], predictions, average="macro", zero_division=0)),
                "training_seconds": elapsed,
            }
        )

    frame = pd.DataFrame(records).sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False
    )
    frame.to_csv(destination / "feature_comparison.csv", index=False)
    return str(frame.iloc[0]["feature_set"])


def tune_model(
    model_name: str,
    feature_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    results_dir: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    balance: bool = True,
    quick: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select hyperparameters exclusively on the validation set."""
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    X_train, y_train = _training_view(train, balance)

    grid = QUICK_MODEL_GRIDS[model_name] if quick else MODEL_GRIDS[model_name]
    records: list[dict[str, object]] = []

    for candidate_number, parameters in enumerate(grid, start=1):
        pipeline = build_pipeline(
            model_name,
            feature_name,
            cache_dir=Path(cache_dir) / f"{model_name}_{candidate_number}",
            quick=quick,
        )
        pipeline.set_params(**parameters)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(validation["text"])
        elapsed = time.perf_counter() - started

        records.append(
            {
                "candidate": candidate_number,
                "parameters": json.dumps(parameters, ensure_ascii=False, default=list),
                "validation_accuracy": float(accuracy_score(validation["label"], predictions)),
                "validation_macro_f1": float(f1_score(validation["label"], predictions, average="macro", zero_division=0)),
                "training_seconds": elapsed,
            }
        )

    frame = pd.DataFrame(records).sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False
    )
    frame.to_csv(destination / f"{model_name}_validation_tuning.csv", index=False)
    best_candidate = int(frame.iloc[0]["candidate"])
    return dict(grid[best_candidate - 1]), frame


def train_and_evaluate_all(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_name: str,
    results_dir: str | Path,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    balance: bool = True,
    quick: bool = False,
) -> list[dict[str, object]]:
    """Tune each required classifier, refit on 80%, and test once on 20%."""
    results_path = Path(results_dir)
    models_path = Path(models_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    training_plus_validation = pd.concat([train, validation], ignore_index=True)
    final_X, final_y = _training_view(training_plus_validation, balance)
    comparison: list[dict[str, object]] = []

    for model_name in ("naive_bayes", "decision_tree", "random_forest", "neural_network"):
        print(f"\nTuning {model_name}...")
        best_parameters, _ = tune_model(
            model_name=model_name,
            feature_name=feature_name,
            train=train,
            validation=validation,
            results_dir=results_path,
            cache_dir=cache_dir,
            balance=balance,
            quick=quick,
        )

        final_pipeline = build_pipeline(
            model_name,
            feature_name,
            cache_dir=Path(cache_dir) / f"final_{model_name}",
            quick=quick,
        )
        final_pipeline.set_params(**best_parameters)

        started = time.perf_counter()
        final_pipeline.fit(final_X, final_y)
        training_seconds = time.perf_counter() - started

        prediction_started = time.perf_counter()
        predictions = final_pipeline.predict(test["text"])
        prediction_seconds = time.perf_counter() - prediction_started

        metrics = save_model_evaluation(
            model_name=model_name,
            y_true=test["label"],
            y_pred=predictions,
            output_dir=results_path,
        )

        artifact = {
            "model_name": model_name,
            "feature_set": feature_name,
            "best_parameters": best_parameters,
            "balance_training": balance,
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "test_metrics": {
                key: metrics[key]
                for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
            },
        }
        (results_path / f"{model_name}_experiment.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, default=list),
            encoding="utf-8",
        )
        joblib.dump(final_pipeline, models_path / f"{model_name}.joblib")

        comparison.append(
            {
                "model": model_name,
                "feature_set": feature_name,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
            }
        )
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    save_model_comparison(comparison, results_path)
    best_model = max(comparison, key=lambda record: float(record["macro_f1"]))
    (results_path / "best_model.json").write_text(
        json.dumps(best_model, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison
