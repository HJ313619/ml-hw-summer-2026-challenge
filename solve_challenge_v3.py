"""
Sofia ML Regression 2026 Summer - Kaggle Challenge (v3)
=========================================================
Submission #3: small hedge against feature-selection variance.

v1: all 20 features + RidgeCV                    -> Public R^2 = 0.57085
v2: top 14 features (by |coef|) + RidgeCV        -> Public R^2 = 0.57092

Extensive testing (different imputers, scalers, robust regression,
automatic feature selection, interaction terms, bagging) all showed
either no improvement or worse CV R^2 than plain top-14 RidgeCV. The
data looks like a clean linear generative process (near-zero feature
correlations, interaction terms actively hurt) with 14 real signal
features and 6 pure-noise features.

v3 change: rather than committing to exactly 14 features, average the
predictions of three RidgeCV models built on the top 13, 14, and 15
features (by |coefficient|). This hedges against the exact cutoff
being slightly off due to sampling noise in which features look
"significant". CV gain over top-14-only is tiny (within noise:
0.5385 vs 0.5384), so this is a low-risk, not-guaranteed-to-help
tweak -- but it doesn't meaningfully differ from v2's core approach.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "summer2026_kaggle_linear_regression_challenge_train.csv"
TEST_PATH = "summer2026_kaggle_linear_regression_challenge_test.csv"
SAMPLE_SUB_PATH = "summer2026_kaggle_linear_regression_challenge_sampleSubmission.csv"
OUTPUT_PATH = "submission_v3.csv"

# Features ranked by |coefficient| from a full-feature RidgeCV fit
# (confirmed stable via 200-sample bootstrap).
FEATURE_RANKING = [
    "x17", "x16", "x0", "x4", "x13", "x2", "x6",
    "x18", "x14", "x19", "x7", "x10", "x15", "x3",
    "x12", "x8", "x1", "x5", "x9", "x11",
]

SUBSET_SIZES = [13, 14, 15]


def make_pipeline():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    y_train = train["target"].values
    test_ids = test["Id"].values

    # --- CV sanity check of the ensemble-of-subset-sizes approach ---
    rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=1)
    fold_scores = []
    for tr_idx, val_idx in rkf.split(train[FEATURE_RANKING].values):
        preds_val = []
        for n in SUBSET_SIZES:
            cols = FEATURE_RANKING[:n]
            Xtr = train[cols].values[tr_idx]
            Xval = train[cols].values[val_idx]
            ytr = y_train[tr_idx]
            pipe = make_pipeline()
            pipe.fit(Xtr, ytr)
            preds_val.append(pipe.predict(Xval))
        pred_avg = np.mean(preds_val, axis=0)
        from sklearn.metrics import r2_score
        fold_scores.append(r2_score(y_train[val_idx], pred_avg))
    print(f"CV R^2 (avg of {SUBSET_SIZES}-feature models): "
          f"mean={np.mean(fold_scores):.5f} std={np.std(fold_scores):.4f}")

    # --- Fit each subset-size model on FULL training data, predict test ---
    test_preds = []
    for n in SUBSET_SIZES:
        cols = FEATURE_RANKING[:n]
        pipe = make_pipeline()
        pipe.fit(train[cols].values, y_train)
        test_preds.append(pipe.predict(test[cols].values))

    final_preds = np.mean(test_preds, axis=0)

    submission = pd.DataFrame({"Id": test_ids, "target": final_preds})
    assert list(submission.columns) == list(sample_sub.columns)
    assert len(submission) == len(sample_sub)

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission written to: {OUTPUT_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
