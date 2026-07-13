# Arabic Sentiment Analysis

A complete machine-learning project for classifying Arabic social-media text into three sentiment classes:

- **Positive**
- **Negative**
- **Objective / Neutral**

This project was developed for the **ENCS3340 Artificial Intelligence** course at **Birzeit University**. It follows the full machine-learning workflow required by the course specification, from dataset analysis and Arabic text preprocessing to model training, evaluation, and prediction.

---

## Project Scope

The repository includes:

- Exploratory analysis of the Arabic social-media dataset
- Arabic text cleaning and normalization
- Stop-word removal, emoji standardization, and optional light stemming
- Word-level TF-IDF features
- Character-level TF-IDF features
- Handcrafted Arabic and social-media features
- Stratified **60% / 20% / 20%** train-validation-test split
- Validation-based feature and hyperparameter selection
- Decision Tree
- Random Forest
- Multinomial Naive Bayes
- Artificial Neural Network
- Accuracy, precision, recall, F1-score, classification reports, and confusion matrices
- Class-imbalance handling using random oversampling on training data only
- Saved trained models
- Command-line prediction demo

---

## Dataset Format

Place the dataset at:

```text
data/raw/arabic_sentiment.txt
```

Each line must follow this format:

```text
Arabic text<TAB>LABEL
```

The parser reads the sentiment label from the final tab using:

```python
text, label = line.rsplit("\t", 1)
```

### Supported Labels

| Original Label | Final Class |
|---|---|
| `POS` | `positive` |
| `NEG` | `negative` |
| `OBJ` | `objective_neutral` |
| `NEUTRAL` | `objective_neutral` |

`OBJ` and `NEUTRAL` are merged into one final class because the project evaluates three sentiment categories.

---

## Methodology

The pipeline follows these steps:

1. Load and validate the tab-separated UTF-8 dataset
2. Detect malformed rows, empty texts, unknown labels, and duplicates
3. Merge `OBJ` and `NEUTRAL` into `objective_neutral`
4. Generate exploratory statistics and visualizations
5. Create a stratified 60% / 20% / 20% split
6. Apply Arabic-aware text preprocessing
7. Compare word, character, and combined TF-IDF features
8. Add handcrafted Arabic and social-media features
9. Tune each required classifier using the validation split
10. Refit the selected configuration on training and validation data
11. Evaluate once on the held-out test set
12. Save trained models and evaluation artifacts

The test set remains isolated until final evaluation to reduce data leakage.

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run the Project

### Full experiment

```bash
python run_all.py
```

### Faster validation run

Uses smaller feature sizes and reduced parameter grids:

```bash
python run_all.py --quick
```

### Select a specific feature representation

```bash
python run_all.py --feature-set combined
```

Supported values depend on the implementation, such as:

```text
word
char
combined
combined_handcrafted
```

### Disable class balancing

```bash
python run_all.py --no-balance
```

---

## Generated Outputs

After execution, the `results/` directory contains generated experiment artifacts such as:

- Dataset summary files
- Original and final class-distribution figures
- Train, validation, and test split files
- Feature-comparison results
- Validation tuning results
- Accuracy, precision, recall, and F1-score tables
- Classification reports
- Confusion matrices
- Model-comparison charts
- Selected best-model metadata

Saved models are written to:

```text
models/
```

Example:

```text
models/naive_bayes.joblib
```

No performance numbers are hardcoded in this README. Final metrics are produced by running the experiment pipeline.

---

## Predict New Arabic Text

After training a model:

```bash
python predict.py --model models/naive_bayes.joblib --text "الخدمة ممتازة جدا"
```

Interactive prediction mode:

```bash
python predict.py --model models/naive_bayes.joblib
```

---

## Class-Imbalance Handling

The dataset contains an imbalanced class distribution.

Random oversampling is applied only to the training data. Validation and test sets remain unchanged so that reported evaluation metrics reflect the original data distribution.

Macro F1-score is used as an important model-selection metric because accuracy alone may hide poor performance on minority classes.

---

## Evaluation Metrics

Each classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The final comparison should consider both overall performance and minority-class performance.

---

## Models

The project includes all model families required by the course:

| Model | Purpose |
|---|---|
| Decision Tree | Interpretable tree-based baseline |
| Random Forest | Ensemble of decision trees with improved stability |
| Multinomial Naive Bayes | Efficient baseline for sparse TF-IDF features |
| Artificial Neural Network | Nonlinear classifier for learned feature representations |

---

## Reproducibility

The project uses:

- A fixed random seed
- Stratified dataset splitting
- Training-only feature fitting
- Training-only oversampling
- Validation-based model selection
- A held-out test set
- Saved model artifacts
- Reproducible command-line scripts

---

## Important Note

The raw dataset should not be uploaded to a public repository unless redistribution is allowed by its original source and university policy.

The source-code license and the dataset usage rights should be treated separately.

---

## Academic Context 
**Arabic Sentiment Analysis Project**
