"""Evaluation utilities for classifiers and experiment comparison.

This module centralizes all test-set evaluation so every classifier is measured
with the same labels, metrics, report format, and visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import FINAL_LABELS


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly version of a model or experiment name."""
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _to_builtin(value: object) -> object:
    """Convert NumPy/Pandas scalar-like values into JSON-compatible objects."""
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return]
    return value


def calculate_metrics(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
) -> dict[str, object]:
    """Calculate the required global and per-class classification metrics."""
    labels = list(FINAL_LABELS)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": report,
    }


def _save_classification_report(
    report: Mapping[str, object],
    destination: Path,
) -> None:
    """Save the classification report as a clean CSV table."""
    rows: list[dict[str, object]] = []

    for label, values in report.items():
        if isinstance(values, Mapping):
            rows.append(
                {
                    "class": label,
                    "precision": float(values.get("precision", 0.0)),
                    "recall": float(values.get("recall", 0.0)),
                    "f1_score": float(values.get("f1-score", 0.0)),
                    "support": int(values.get("support", 0)),
                }
            )

    pd.DataFrame(rows).to_csv(destination, index=False, encoding="utf-8-sig")


def _save_confusion_matrix(
    model_name: str,
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    output_dir: Path,
) -> None:
    """Save raw/normalized confusion matrices and their visualizations."""
    labels = list(FINAL_LABELS)
    safe_model_name = _safe_name(model_name)

    raw_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    raw_frame = pd.DataFrame(raw_matrix, index=labels, columns=labels)
    raw_frame.index.name = "actual"
    raw_frame.columns.name = "predicted"
    raw_frame.to_csv(
        output_dir / f"{safe_model_name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    normalized_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )
    normalized_frame = pd.DataFrame(normalized_matrix, index=labels, columns=labels)
    normalized_frame.index.name = "actual"
    normalized_frame.columns.name = "predicted"
    normalized_frame.to_csv(
        output_dir / f"{safe_model_name}_confusion_matrix_normalized.csv",
        encoding="utf-8-sig",
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=raw_matrix, display_labels=labels)
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title(f"Confusion Matrix — {model_name}")
    figure.tight_layout()
    figure.savefig(output_dir / f"{safe_model_name}_confusion_matrix.png", dpi=200)
    plt.close(figure)

    normalized_figure, normalized_axis = plt.subplots(figsize=(8, 6))
    normalized_display = ConfusionMatrixDisplay(
        confusion_matrix=normalized_matrix,
        display_labels=labels,
    )
    normalized_display.plot(ax=normalized_axis, values_format=".2f", colorbar=False)
    normalized_axis.set_title(f"Normalized Confusion Matrix — {model_name}")
    normalized_figure.tight_layout()
    normalized_figure.savefig(
        output_dir / f"{safe_model_name}_confusion_matrix_normalized.png",
        dpi=200,
    )
    plt.close(normalized_figure)


def save_model_evaluation(
    model_name: str,
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    output_dir: str | Path,
) -> dict[str, object]:
    """Evaluate one classifier and save all report artifacts.

    Returns the metric dictionary so the training workflow can construct a
    comparison table without recalculating the results.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    safe_model_name = _safe_name(model_name)

    metrics = calculate_metrics(y_true, y_pred)
    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "classification_report"
    }
    (destination / f"{safe_model_name}_metrics.json").write_text(
        json.dumps(serializable_metrics, ensure_ascii=False, indent=2, default=_to_builtin),
        encoding="utf-8",
    )

    report = metrics["classification_report"]
    if not isinstance(report, Mapping):
        raise TypeError("classification_report must be a mapping")

    _save_classification_report(
        report,
        destination / f"{safe_model_name}_classification_report.csv",
    )
    _save_confusion_matrix(model_name, y_true, y_pred, destination)

    predictions_frame = pd.DataFrame(
        {
            "actual_label": list(y_true),
            "predicted_label": list(y_pred),
        }
    )
    predictions_frame["is_correct"] = (
        predictions_frame["actual_label"] == predictions_frame["predicted_label"]
    )
    predictions_frame.to_csv(
        destination / f"{safe_model_name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metrics


def save_model_comparison(
    records: Iterable[Mapping[str, object]],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Save a ranked metric table and comparison figures for all classifiers."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("At least one model result is required for comparison")

    required_columns = {
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Missing model comparison columns: " + ", ".join(sorted(missing_columns))
        )

    frame = frame.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.to_csv(
        destination / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]
    axis = frame.set_index("model")[metric_columns].plot(
        kind="bar",
        figsize=(11, 6),
    )
    axis.set_title("Classifier Performance Comparison")
    axis.set_xlabel("Classifier")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.legend(title="Metric", loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(destination / "model_comparison.png", dpi=200)
    plt.close(axis.figure)

    if {"training_seconds", "prediction_seconds"}.issubset(frame.columns):
        timing_axis = frame.set_index("model")[["training_seconds", "prediction_seconds"]].plot(
            kind="bar",
            figsize=(10, 6),
        )
        timing_axis.set_title("Training and Prediction Time")
        timing_axis.set_xlabel("Classifier")
        timing_axis.set_ylabel("Seconds")
        timing_axis.legend(title="Timing")
        timing_axis.figure.tight_layout()
        timing_axis.figure.savefig(destination / "model_timing_comparison.png", dpi=200)
        plt.close(timing_axis.figure)

    return frame
