"""Central configuration for the Arabic sentiment analysis project."""

from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "arabic_sentiment.txt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"

ORIGINAL_LABELS = ("POS", "NEG", "OBJ", "NEUTRAL")
LABEL_MAP = {
    "POS": "positive",
    "NEG": "negative",
    "OBJ": "objective_neutral",
    "NEUTRAL": "objective_neutral",
}
FINAL_LABELS = ("negative", "objective_neutral", "positive")

# Negation words are deliberately NOT removed as stop words because they can
# reverse sentiment polarity, e.g. "مش حلو".
NEGATION_WORDS = {
    "لا", "ليس", "ليست", "لم", "لن", "ما", "مش", "مو", "مب", "بدون", "غير",
}

# Compact built-in list so the project runs without downloading NLTK corpora.
ARABIC_STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "هو", "هي", "هم", "هن", "انا", "أنا", "انت", "أنت", "انتم", "نحن", "كان",
    "كانت", "يكون", "تكون", "وقد", "قد", "ثم", "او", "أو", "و", "ف", "ب", "ك",
    "ل", "الذي", "التي", "الذين", "اللاتي", "كل", "أي", "اي", "هناك", "هنا",
    "بعد", "قبل", "حتى", "إذا", "اذا", "إن", "ان", "أن", "كما", "لكن", "بل",
    "أيضا", "ايضا", "بين", "عند", "ضمن", "حول", "أمام", "امام", "خلال", "منذ",
    "أكثر", "اكثر", "أقل", "اقل", "مرة", "جدا", "جداً", "تم", "يمكن", "لقد",
    "ولا", "فقط", "اليوم", "غدا", "غداً", "الآن", "الان", "كنت", "كانت", "يتم",
} - NEGATION_WORDS

POSITIVE_EMOJIS = set("😀😃😄😁😊😍🥰😘😋😎🙂☺❤♥💚💙💜💛✨🎉👍👏🌹")
NEGATIVE_EMOJIS = set("😞😔😢😭😡😠🤬😣😖😫😩😒🙁☹💔👎😕😟")

FEATURE_CONFIGS = {
    "word": {
        "description": "Word TF-IDF with unigrams and bigrams",
        "word": True,
        "char": False,
        "handcrafted": False,
    },
    "char": {
        "description": "Character TF-IDF for spelling and dialect variation",
        "word": False,
        "char": True,
        "handcrafted": False,
    },
    "combined": {
        "description": "Word TF-IDF + character TF-IDF + handcrafted Arabic features",
        "word": True,
        "char": True,
        "handcrafted": True,
    },
}

MODEL_GRIDS = {
    "naive_bayes": [
        {"classifier__alpha": 0.1},
        {"classifier__alpha": 0.5},
        {"classifier__alpha": 1.0},
    ],
    "decision_tree": [
        {"classifier__max_depth": 20, "classifier__min_samples_split": 5, "classifier__class_weight": None},
        {"classifier__max_depth": 30, "classifier__min_samples_split": 5, "classifier__class_weight": "balanced"},
        {"classifier__max_depth": 50, "classifier__min_samples_split": 10, "classifier__class_weight": "balanced"},
        {"classifier__max_depth": None, "classifier__min_samples_split": 10, "classifier__class_weight": "balanced"},
    ],
    "random_forest": [
        {"classifier__n_estimators": 150, "classifier__max_depth": 25, "classifier__min_samples_leaf": 1, "classifier__class_weight": None},
        {"classifier__n_estimators": 200, "classifier__max_depth": 35, "classifier__min_samples_leaf": 1, "classifier__class_weight": "balanced"},
        {"classifier__n_estimators": 250, "classifier__max_depth": None, "classifier__min_samples_leaf": 1, "classifier__class_weight": "balanced"},
        {"classifier__n_estimators": 200, "classifier__max_depth": None, "classifier__min_samples_leaf": 2, "classifier__class_weight": "balanced_subsample"},
    ],
    "neural_network": [
        {"classifier__hidden_layer_sizes": (128,), "classifier__activation": "relu", "classifier__learning_rate_init": 0.001},
        {"classifier__hidden_layer_sizes": (128, 64), "classifier__activation": "relu", "classifier__learning_rate_init": 0.001},
        {"classifier__hidden_layer_sizes": (256, 128), "classifier__activation": "relu", "classifier__learning_rate_init": 0.0005},
        {"classifier__hidden_layer_sizes": (128, 64), "classifier__activation": "tanh", "classifier__learning_rate_init": 0.001},
    ],
}

QUICK_MODEL_GRIDS = {
    "naive_bayes": [{"classifier__alpha": 0.5}],
    "decision_tree": [{"classifier__max_depth": 20, "classifier__min_samples_split": 5, "classifier__class_weight": "balanced"}],
    "random_forest": [{"classifier__n_estimators": 60, "classifier__max_depth": 25, "classifier__min_samples_leaf": 1, "classifier__class_weight": "balanced"}],
    "neural_network": [{"classifier__hidden_layer_sizes": (32,), "classifier__activation": "relu", "classifier__learning_rate_init": 0.001}],
}
