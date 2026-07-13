# Execution Results

## Run configuration

The complete four-model result set was generated with:

```bash
python run_all.py --quick --no-balance --feature-set combined
```

- Dataset records: 10,006
- Train: 6,003
- Validation: 2,001
- Test: 2,002
- Feature set: Word TF-IDF + Character TF-IDF + handcrafted features
- Random state: 42
- Random oversampling: disabled for this completed run

## Model comparison

| Rank | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Naive Bayes | 0.755744 | 0.578761 | 0.440988 | **0.466877** | 0.715932 |
| 2 | Random Forest | 0.692308 | 0.464116 | 0.445737 | 0.453335 | 0.682388 |
| 3 | Decision Tree | 0.506494 | 0.401088 | 0.443260 | 0.388786 | 0.549430 |
| 4 | Neural Network | **0.760739** | 0.711474 | 0.372843 | 0.364211 | 0.683039 |

The selected model by Macro F1 is **Naive Bayes**.

## Notes

- The Neural Network achieved the highest accuracy, but it predicted the majority objective/neutral class too often, resulting in a low Macro F1.
- Naive Bayes provided the best balance across the three classes according to Macro F1.
- The dataset is strongly imbalanced, so Macro F1 is more informative than accuracy alone.
- The quick neural-network configuration reached the maximum of 10 iterations without convergence; its score should be treated as a fast-run result rather than a final optimized ANN result.

## Generated artifacts

The results folder includes:

- Dataset summaries and EDA figures
- Train, validation, and test split CSV files
- Validation tuning CSV files
- Per-model metrics JSON files
- Classification reports
- Raw and normalized confusion matrices
- Prediction CSV files
- Model comparison figures and CSV
- Saved `.joblib` models
