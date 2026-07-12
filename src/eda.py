"""Exploratory data analysis and required visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing import ArabicTextPreprocessor


def _save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=axis)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_eda(data: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    """Generate dataset summaries, class charts, and frequent-term tables."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    original_counts = data["original_label"].value_counts().sort_index()
    merged_counts = data["label"].value_counts().sort_index()

    summary = {
        "total_samples": int(len(data)),
        "empty_texts": int(data["text"].str.strip().eq("").sum()),
        "duplicate_texts": int(data["text"].duplicated().sum()),
        "original_class_distribution": {key: int(value) for key, value in original_counts.items()},
        "merged_class_distribution": {key: int(value) for key, value in merged_counts.items()},
        "text_length": {
            "mean": float(data["text_length"].mean()),
            "median": float(data["text_length"].median()),
            "minimum": int(data["text_length"].min()),
            "maximum": int(data["text_length"].max()),
        },
        "word_count": {
            "mean": float(data["word_count"].mean()),
            "median": float(data["word_count"].median()),
            "minimum": int(data["word_count"].min()),
            "maximum": int(data["word_count"].max()),
        },
    }

    (destination / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    original_counts.rename("count").to_csv(destination / "original_class_distribution.csv")
    merged_counts.rename("count").to_csv(destination / "merged_class_distribution.csv")

    _save_bar(
        original_counts,
        "Original Sentiment Class Distribution",
        "Original label",
        "Number of texts",
        destination / "original_class_distribution.png",
    )
    _save_bar(
        merged_counts,
        "Final Three-Class Distribution",
        "Final label",
        "Number of texts",
        destination / "final_class_distribution.png",
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(data["text_length"], bins=40)
    axis.set_title("Arabic Text Length Distribution")
    axis.set_xlabel("Number of characters")
    axis.set_ylabel("Frequency")
    figure.tight_layout()
    figure.savefig(destination / "text_length_distribution.png", dpi=180)
    plt.close(figure)

    preprocessor = ArabicTextPreprocessor(remove_stopwords=True, stemming="light")
    top_word_records: list[dict[str, object]] = []
    for label, group in data.groupby("label", sort=True):
        vectorizer = CountVectorizer(
            preprocessor=preprocessor,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 1),
            min_df=2,
            max_features=20,
        )
        matrix = vectorizer.fit_transform(group["text"])
        totals = matrix.sum(axis=0).A1
        pairs = sorted(
            zip(vectorizer.get_feature_names_out(), totals),
            key=lambda item: item[1],
            reverse=True,
        )
        for rank, (term, count) in enumerate(pairs[:20], start=1):
            top_word_records.append(
                {"label": label, "rank": rank, "term": term, "count": int(count)}
            )

    pd.DataFrame(top_word_records).to_csv(destination / "top_words_by_class.csv", index=False)
    return summary
