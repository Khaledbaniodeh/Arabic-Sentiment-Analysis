#!/usr/bin/env python3
"""Run the complete university project from EDA to final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    FEATURE_CONFIGS,
)
from src.data_loader import load_dataset, split_dataset
from src.eda import run_eda
from src.training import select_feature_set, train_and_evaluate_all


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arabic sentiment analysis experiments")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--feature-set",
        choices=["auto", *FEATURE_CONFIGS.keys()],
        default="auto",
        help="auto compares word, char, and combined features on validation data",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="disable random oversampling of the training data",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use small grids and fewer features for a fast code check",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.results.mkdir(parents=True, exist_ok=True)
    arguments.models.mkdir(parents=True, exist_ok=True)
    arguments.cache.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {arguments.data}")
    data = load_dataset(arguments.data)
    print(f"Valid samples: {len(data)}")
    print("Final label distribution:")
    print(data["label"].value_counts())

    print("\nRunning exploratory data analysis...")
    summary = run_eda(data, arguments.results / "eda")

    train, validation, test = split_dataset(data)
    split_summary = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    (arguments.results / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )
    train.to_csv(arguments.results / "train_split.csv", index=False)
    validation.to_csv(arguments.results / "validation_split.csv", index=False)
    test.to_csv(arguments.results / "test_split.csv", index=False)
    print(f"Split sizes: {split_summary}")

    balance = not arguments.no_balance
    if arguments.feature_set == "auto":
        print("\nComparing feature extraction methods...")
        selected_feature = select_feature_set(
            train=train,
            validation=validation,
            results_dir=arguments.results,
            cache_dir=arguments.cache,
            balance=balance,
            quick=arguments.quick,
        )
    else:
        selected_feature = arguments.feature_set

    print(f"Selected feature set: {selected_feature}")
    (arguments.results / "selected_feature_set.json").write_text(
        json.dumps(
            {
                "feature_set": selected_feature,
                "description": FEATURE_CONFIGS[selected_feature]["description"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison = train_and_evaluate_all(
        train=train,
        validation=validation,
        test=test,
        feature_name=selected_feature,
        results_dir=arguments.results,
        models_dir=arguments.models,
        cache_dir=arguments.cache,
        balance=balance,
        quick=arguments.quick,
    )

    print("\nCompleted successfully.")
    print(f"Results folder: {arguments.results}")
    print(f"Models folder: {arguments.models}")
    best = max(comparison, key=lambda item: float(item["macro_f1"]))
    print(f"Best model by Macro F1: {best['model']} ({best['macro_f1']:.4f})")


if __name__ == "__main__":
    main()
