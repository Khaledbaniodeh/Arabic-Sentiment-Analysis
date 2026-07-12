#!/usr/bin/env python3
"""Predict the sentiment of new Arabic text using a saved model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Arabic sentiment")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .joblib model")
    parser.add_argument("--text", type=str, help="Arabic text to classify")
    return parser.parse_args()


def predict(model_path: Path, text: str) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    pipeline = joblib.load(model_path)
    label = pipeline.predict([text])[0]
    print(f"Predicted label: {label}")

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        order = np.argsort(probabilities)[::-1]
        print("Probabilities:")
        for index in order:
            print(f"  {classes[index]}: {probabilities[index]:.4f}")


def main() -> None:
    arguments = parse_arguments()
    if arguments.text:
        predict(arguments.model, arguments.text)
        return

    print("Interactive mode. Type 'exit' to stop.")
    while True:
        text = input("Arabic text: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            predict(arguments.model, text)


if __name__ == "__main__":
    main()
