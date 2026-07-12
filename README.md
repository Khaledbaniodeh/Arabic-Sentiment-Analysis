# Arabic Sentiment Analysis

A complete, reproducible ENCS3340 machine-learning project for classifying Arabic social-media text into **negative**, **neutral/objective**, and **positive** sentiment.

## Highlights
- Robust UTF-8 loader with validation and duplicate detection.
- `OBJ` and `NEUTRAL` are merged into the final `neutral` class while preserving original labels.
- Leakage-safe stratified 60/20/20 split (`random_state=42`).
- Word TF-IDF, character TF-IDF, and Arabic/social-media handcrafted features.
- Required models: Decision Tree, Random Forest, Complement Naive Bayes, and ANN (MLP).
- Real EDA figures, metrics, confusion matrix, saved models, tests, CLI, and Streamlit demo.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/run_eda.py
python scripts/train.py
python scripts/predict.py "هذا المنتج رائع جدا"
streamlit run app/streamlit_app.py
```

## Methodology
Exact duplicates are removed before splitting. TF-IDF, SVD, and scaling are inside scikit-learn pipelines and therefore fitted only on training data. Validation Macro F1 selects the final model. The held-out test split is evaluated once after selection.

## Repository structure
- `src/arabic_sentiment/`: reusable implementation
- `scripts/`: EDA, training, inference
- `tests/`: unit tests
- `reports/`: generated figures, results, and formal PDF
- `models/`: trained pipelines
- `app/`: Streamlit demo
- `docs/`: design, status, and defense notes
