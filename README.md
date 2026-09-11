# Sofia ML Regression 2026 Summer — Kaggle Challenge

A linear regression solution to the Sofia University ML course's Kaggle competition: predicting a real-valued `target` from 20 anonymized numeric features.

- **Competition:** [Sofia ML Regression 2026 Summer](https://www.kaggle.com/competitions/sofia-ml-regression-2026-summer-private)
- **Metric:** R² (coefficient of determination), matching scikit-learn's `model.score(testX, testY)`
- **Final public R²:** 0.57124
- **Final private R²:** 0.54018

## Overview

The task looked like a generic regression problem, but exploratory analysis pointed to one clear story: a linear relationship hidden in 20 features, only 14 of which carry real signal — the other 6 are statistically indistinguishable from noise.

The final pipeline:

```
median impute → standard scale → RidgeCV, averaged across 13/14/15-feature subsets
```

## Files

| File | Description |
|---|---|
| `solve_challenge.py` | **v1** — baseline: all 20 features + RidgeCV |
| `solve_challenge_v2.py` | **v2** — top-14 features (by \|coefficient\|) + RidgeCV |
| `solve_challenge_v3.py` | **v3 (final)** — averages RidgeCV models fit on the top 13, 14, and 15 features, hedging against the exact feature-selection cutoff |

Each script:
1. Loads `train.csv` / `test.csv` / `sampleSubmission.csv`
2. Median-imputes missing values (~1% per feature)
3. Standard-scales features
4. Fits `RidgeCV` (regularization strength chosen automatically via internal CV)
5. Cross-validates with repeated K-fold to report an honest R² estimate
6. Predicts on the test set and writes a submission CSV matching the required format

## Results

| Version | Approach | CV R² | Public R² |
|---|---|---|---|
| v1 | All 20 features + RidgeCV | 0.5366 | 0.57085 |
| v2 | Top-14 features + RidgeCV | 0.5387 | 0.57092 |
| **v3** | **Avg. of 13/14/15-feature RidgeCV models** | **0.5385** | **0.57124** |

**Final private R² (v3): 0.54018**

Notably, the cross-validated estimate (0.5387) landed just 0.0015 away from the true private score (0.54018) — much closer than the public leaderboard score (0.57124), which overstated performance by about 0.03. This validates using CV rather than the public leaderboard to guide model decisions.

## What worked / what didn't

**Worked:**
- Dropping the 6 noise features (bootstrap-confirmed their coefficients were indistinguishable from zero)
- RidgeCV's automatic regularization (mainly damping the noisy features, since real features were already near-uncorrelated)
- Averaging across the 13/14/15-feature cutoff (hedging against selection uncertainty)
- Trusting cross-validation over the public leaderboard score

**Didn't work:**
- Polynomial/interaction terms (hurt CV R² — confirms the relationship is genuinely linear)
- Robust regression (Huber) — underperformed plain Ridge since there were no real outliers
- Automatic feature selection (Lasso/ElasticNet) — slightly worse than manual top-14 selection
- Bagging / model ensembling — no gain, since the base linear model was already low-variance
- Alternative imputers (KNN, iterative) and scalers (Robust, MinMax) — no meaningful difference

## Requirements

```
numpy
pandas
scikit-learn
```

## Usage

Place `train.csv`, `test.csv`, and `sampleSubmission.csv` (named per the competition's file names) in the same directory as the script, then run:

```bash
python solve_challenge_v3.py
```

This prints the cross-validated R² estimate and writes `submission_v3.csv` in the Kaggle-required format (`Id,target`).
