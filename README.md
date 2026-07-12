# Arabic Sentiment Analysis


## Project scope

The code implements the university requirements only:

- Exploratory analysis of the Arabic social-media dataset.
- Three final classes: `positive`, `negative`, and `objective_neutral`.
- Arabic cleaning, normalization, stop-word removal, emoji standardization, and light stemming.
- Word TF-IDF, character TF-IDF, and handcrafted Arabic/social-media features.
- A stratified `60% / 20% / 20%` train-validation-test split.
- Validation-based hyperparameter selection.
- Decision Tree, Random Forest, Multinomial Naive Bayes, and Artificial Neural Network.
- Accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
- Class-imbalance handling through random oversampling of training data only.
- Saved models and a small prediction demo.

## Dataset format

The included dataset is stored at:

```text
data/raw/arabic_sentiment.txt
```

Each line has this format:

```text
Arabic text<TAB>LABEL
```

Supported original labels:

- `POS` → `positive`
- `NEG` → `negative`
- `OBJ` → `objective_neutral`
- `NEUTRAL` → `objective_neutral`

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the complete project

Full experiment:

```bash
python run_all.py
```

Fast code check with smaller feature sizes and parameter grids:

```bash
python run_all.py --quick
```

Use a specific feature representation instead of automatic comparison:

```bash
python run_all.py --feature-set combined
```

Disable training oversampling:

```bash
python run_all.py --no-balance
```

## Outputs

The `results/` directory will contain:

- Dataset summaries and class-distribution charts.
- Train, validation, and test split files.
- Feature-method comparison.
- Validation tuning results for every classifier.
- Accuracy, precision, recall, F1-score, and classification reports.
- Confusion matrices and model-comparison charts.
- The selected best model information.

The trained models are stored in `models/` as `.joblib` files.

## Predict new Arabic text

After training:

```bash
python predict.py --model models/naive_bayes.joblib --text "الخدمة ممتازة جدا"
```

Interactive mode:

```bash
python predict.py --model models/naive_bayes.joblib
```

## Methodology

1. Load and validate the tab-separated dataset.
2. Merge `OBJ` and `NEUTRAL` into `objective_neutral` as required by the project specification.
3. Generate exploratory statistics and figures.
4. Make a stratified 60/20/20 split.
5. Compare word TF-IDF, character TF-IDF, and combined features on validation data.
6. Tune each required classifier on the validation split.
7. Refit the selected setting on training + validation data.
8. Evaluate exactly once on the held-out test set.

## Important note

Oversampling is applied only to training data. Validation and test data remain untouched so evaluation remains valid.
