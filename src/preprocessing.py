"""Arabic text cleaning and normalization."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from nltk.stem.isri import ISRIStemmer

from .config import (
    ARABIC_STOPWORDS,
    NEGATION_WORDS,
    NEGATIVE_EMOJIS,
    POSITIVE_EMOJIS,
)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#([^\s#]+)")
EMAIL_RE = re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NUMBER_RE = re.compile(r"[0-9٠-٩]+")
DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_OR_LATIN_RE = re.compile(r"[^\u0600-\u06FFA-Za-z_\s]")
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


@dataclass(frozen=True)
class PreprocessingOptions:
    remove_stopwords: bool = True
    stemming: str = "light"  # none, light, isri
    preserve_emoji_tokens: bool = True
    preserve_hashtag_words: bool = True


_ISRI = ISRIStemmer()


def normalize_arabic(text: str) -> str:
    """Normalize common Arabic letter variants and remove diacritics."""
    text = DIACRITICS_RE.sub("", text)
    text = text.replace("ـ", "")
    translation = str.maketrans(
        {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ى": "ي",
            "ؤ": "و",
            "ئ": "ي",
            "ة": "ه",
            "گ": "ك",
            "ڤ": "ف",
            "پ": "ب",
        }
    )
    return text.translate(translation)


def light_stem_arabic(token: str) -> str:
    """Apply conservative Arabic prefix/suffix stripping.

    This is intentionally lighter than root stemming to preserve sentiment words.
    """
    if len(token) <= 3 or token in NEGATION_WORDS:
        return token

    prefixes = ("وال", "بال", "كال", "فال", "لل", "ال")
    suffixes = ("يات", "ات", "ون", "ين", "ان", "يه", "ها", "هم", "هن", "كم", "كن", "نا", "ه", "ي")

    result = token
    for prefix in prefixes:
        if result.startswith(prefix) and len(result) - len(prefix) >= 3:
            result = result[len(prefix):]
            break

    for suffix in suffixes:
        if result.endswith(suffix) and len(result) - len(suffix) >= 3:
            result = result[:-len(suffix)]
            break
    return result


def _emoji_tokens(text: str) -> tuple[str, int, int]:
    positive_count = sum(text.count(symbol) for symbol in POSITIVE_EMOJIS)
    negative_count = sum(text.count(symbol) for symbol in NEGATIVE_EMOJIS)

    for symbol in POSITIVE_EMOJIS:
        text = text.replace(symbol, " ايموجي_ايجابي ")
    for symbol in NEGATIVE_EMOJIS:
        text = text.replace(symbol, " ايموجي_سلبي ")
    return text, positive_count, negative_count


def preprocess_arabic_text(
    text: object,
    options: PreprocessingOptions | None = None,
) -> str:
    """Clean and normalize a single Arabic social-media text."""
    options = options or PreprocessingOptions()
    value = "" if text is None else str(text)
    value = html.unescape(value).lower()
    value = HTML_TAG_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    value = EMAIL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)

    if options.preserve_hashtag_words:
        value = HASHTAG_RE.sub(lambda match: " " + match.group(1).replace("_", " ") + " ", value)
    else:
        value = HASHTAG_RE.sub(" ", value)

    if options.preserve_emoji_tokens:
        value, _, _ = _emoji_tokens(value)

    value = NUMBER_RE.sub(" ", value)
    value = normalize_arabic(value)
    value = REPEATED_CHAR_RE.sub(r"\1\1", value)
    value = ARABIC_OR_LATIN_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip()

    tokens = value.split()
    if options.remove_stopwords:
        tokens = [token for token in tokens if token not in ARABIC_STOPWORDS]

    if options.stemming == "light":
        tokens = [light_stem_arabic(token) for token in tokens]
    elif options.stemming == "isri":
        tokens = [
            token if token in NEGATION_WORDS or not re.search(r"[\u0600-\u06FF]", token)
            else _ISRI.stem(token)
            for token in tokens
        ]
    elif options.stemming != "none":
        raise ValueError("stemming must be one of: none, light, isri")

    return " ".join(token for token in tokens if token)


class ArabicTextPreprocessor:
    """Pickle-safe callable for scikit-learn vectorizers."""

    def __init__(
        self,
        remove_stopwords: bool = True,
        stemming: str = "light",
        preserve_emoji_tokens: bool = True,
        preserve_hashtag_words: bool = True,
    ) -> None:
        self.options = PreprocessingOptions(
            remove_stopwords=remove_stopwords,
            stemming=stemming,
            preserve_emoji_tokens=preserve_emoji_tokens,
            preserve_hashtag_words=preserve_hashtag_words,
        )

    def __call__(self, text: object) -> str:
        return preprocess_arabic_text(text, self.options)
