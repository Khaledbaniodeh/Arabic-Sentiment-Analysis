"""TF-IDF and handcrafted feature construction."""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import FeatureUnion

from .config import FEATURE_CONFIGS, NEGATION_WORDS, NEGATIVE_EMOJIS, POSITIVE_EMOJIS
from .preprocessing import ArabicTextPreprocessor

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#[^\s#]+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
ARABIC_WORD_RE = re.compile(r"[\u0600-\u06FF]+")


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Extract non-negative linguistic and social-media features."""

    feature_names = (
        "log_character_count",
        "log_word_count",
        "average_word_length",
        "exclamation_count",
        "question_count",
        "hashtag_count",
        "mention_count",
        "url_count",
        "elongation_count",
        "positive_emoji_count",
        "negative_emoji_count",
        "negation_count",
        "arabic_word_ratio",
    )

    def fit(self, X: Iterable[object], y: object = None) -> "HandcraftedFeatures":
        return self

    def transform(self, X: Iterable[object]) -> sparse.csr_matrix:
        rows: list[list[float]] = []
        for item in X:
            text = "" if item is None else str(item)
            words = text.split()
            arabic_words = ARABIC_WORD_RE.findall(text)
            total_words = max(len(words), 1)
            average_word_length = sum(len(word) for word in words) / total_words
            negation_count = sum(1 for word in words if word in NEGATION_WORDS)

            rows.append(
                [
                    math.log1p(len(text)),
                    math.log1p(len(words)),
                    min(average_word_length / 20.0, 1.0),
                    math.log1p(text.count("!")),
                    math.log1p(text.count("?")),
                    math.log1p(len(HASHTAG_RE.findall(text))),
                    math.log1p(len(MENTION_RE.findall(text))),
                    math.log1p(len(URL_RE.findall(text))),
                    math.log1p(len(REPEATED_CHAR_RE.findall(text))),
                    math.log1p(sum(text.count(symbol) for symbol in POSITIVE_EMOJIS)),
                    math.log1p(sum(text.count(symbol) for symbol in NEGATIVE_EMOJIS)),
                    math.log1p(negation_count),
                    len(arabic_words) / total_words,
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.feature_names, dtype=object)


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """Select up to k chi-square features without failing on small datasets."""

    def __init__(self, k: int = 3000) -> None:
        self.k = k
        self.selector_: SelectKBest | None = None

    def fit(self, X: sparse.spmatrix, y: Iterable[object]) -> "SafeSelectKBest":
        effective_k = min(self.k, X.shape[1])
        self.selector_ = SelectKBest(score_func=chi2, k=effective_k)
        self.selector_.fit(X, y)
        return self

    def transform(self, X: sparse.spmatrix) -> sparse.spmatrix:
        if self.selector_ is None:
            raise RuntimeError("SafeSelectKBest must be fitted before transform.")
        return self.selector_.transform(X)

    def get_support(self) -> np.ndarray:
        if self.selector_ is None:
            raise RuntimeError("SafeSelectKBest must be fitted before get_support.")
        return self.selector_.get_support()


def build_feature_union(feature_name: str, quick: bool = False) -> FeatureUnion:
    """Build one of the feature representations required by the project."""
    if feature_name not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature set: {feature_name}")

    config = FEATURE_CONFIGS[feature_name]
    transformers: list[tuple[str, object]] = []

    if config["word"]:
        transformers.append(
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=ArabicTextPreprocessor(
                        remove_stopwords=True,
                        stemming="light",
                        preserve_emoji_tokens=True,
                        preserve_hashtag_words=True,
                    ),
                    tokenizer=str.split,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=3000 if quick else 12000,
                    dtype=np.float64,
                ),
            )
        )

    if config["char"]:
        transformers.append(
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=ArabicTextPreprocessor(
                        remove_stopwords=False,
                        stemming="none",
                        preserve_emoji_tokens=True,
                        preserve_hashtag_words=True,
                    ),
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.995,
                    sublinear_tf=True,
                    max_features=2000 if quick else 10000,
                    dtype=np.float64,
                ),
            )
        )

    if config["handcrafted"]:
        transformers.append(("handcrafted", HandcraftedFeatures()))

    return FeatureUnion(transformer_list=transformers, n_jobs=1)
