"""Dataset loading, validation, splitting, and class balancing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LABEL_MAP, ORIGINAL_LABELS, RANDOM_STATE


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the tab-separated Arabic dataset.

    Each non-empty line must end with one of: POS, NEG, OBJ, NEUTRAL.
    The split is performed from the final tab so any tabs inside the text do
    not break parsing.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: list[dict[str, object]] = []
    malformed: list[int] = []

    with dataset_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" not in line:
                malformed.append(line_number)
                continue

            text, original_label = line.rsplit("\t", 1)
            text = text.strip()
            original_label = original_label.strip().upper()

            if not text or original_label not in ORIGINAL_LABELS:
                malformed.append(line_number)
                continue

            rows.append(
                {
                    "text": text,
                    "original_label": original_label,
                    "label": LABEL_MAP[original_label],
                    "line_number": line_number,
                }
            )

    if malformed:
        preview = malformed[:10]
        raise ValueError(
            f"Found {len(malformed)} malformed lines. First line numbers: {preview}"
        )
    if not rows:
        raise ValueError("The dataset contains no valid samples.")

    data = pd.DataFrame(rows)
    data["text_length"] = data["text"].str.len()
    data["word_count"] = data["text"].str.split().str.len()
    return data


def split_dataset(
    data: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a stratified 60% train, 20% validation, 20% test split."""
    required_columns = {"text", "label"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train, temporary = train_test_split(
        data,
        test_size=0.40,
        random_state=random_state,
        stratify=data["label"],
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=random_state,
        stratify=temporary["label"],
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def random_oversample(
    texts: pd.Series,
    labels: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series]:
    """Randomly oversample minority classes using only the training data."""
    frame = pd.DataFrame({"text": texts.astype(str), "label": labels.astype(str)})
    maximum = int(frame["label"].value_counts().max())
    rng = np.random.RandomState(random_state)

    balanced_parts = []
    for _, group in frame.groupby("label", sort=True):
        if len(group) < maximum:
            sampled_indices = rng.choice(group.index.to_numpy(), size=maximum, replace=True)
            balanced_parts.append(group.loc[sampled_indices])
        else:
            balanced_parts.append(group)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced["text"], balanced["label"]
