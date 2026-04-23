# Classification Plan — Value Score Prediction

This document is the design contract for the classification track of The PC
Hardware Auditor project. It is paired with the executable notebook
[notebooks/Classification.ipynb](../../notebooks/Classification.ipynb).

## 1. Why classification, and why multiclass first

Two binding pieces of guidance frame the design:

- The team's own [Categorical Encoding & Validation Strategy.txt](Categorical%20Encoding%20%26%20Validation%20Strategy.txt)
  (section 3) already commits to a four-class Value Score in direct response
  to instructor feedback.
- The latest round of professor comments asks explicitly:
  - "Price can be categorized into multiple categories for classification
    prediction. After converting price to classes, check how imbalance is
    among the classes."
  - "Just wondering why only binary classes?"
  - "Remember we will be discussing more classification methods like SVM,
    SVM RBF, decision tree, GBM. They are much more powerful than logistic.
    If your team builds multiple models (SVM, RBF, GBM), then put them
    together as an additional ensemble model."

Taken together, the classification track is **in scope for the final report**
(not optional), the **primary label is multiclass**, and the **model slate
must include non-linear classifiers plus an ensemble**. The original
binary "good deal vs. not" question becomes a robustness check that
appears as a short sub-section of the report rather than the headline.

## 2. Label definitions (all derived from regression residuals)

Residuals come from `reports/residuals.parquet`, produced by
[notebooks/Project_refactored.ipynb](../../notebooks/Project_refactored.ipynb).
For each row the parquet contains `split` (train/test), `y_true`, `y_pred`,
`residual` (USD), and `residual_std` (z-score using the train-only standard
deviation).

### 2.1 Primary label — Value Score (4 classes)

Cut points are computed on the **train residual distribution only** and then
applied identically to test. Ordered from best-value to worst-value:

| class | label           | definition                                  |
|-------|-----------------|---------------------------------------------|
| 0     | Steal           | `residual` <= 10th train percentile          |
| 1     | Fair Value      | 10th < `residual` <= 50th                    |
| 2     | Brand Premium   | 50th < `residual` <= 90th                    |
| 3     | Extreme Tax     | `residual` > 90th train percentile           |

Expected class proportions by construction are 10 / 40 / 40 / 10 percent on
train, with small deviations on test driven by how well the residual
distribution generalizes.

### 2.2 Robustness binary — Steal vs. not

`y = 1` if the row is in class 0 (Steal), else `y = 0`. This is the strongest
"good deal vs. not" framing that does not conflate "cheap" with "good
value."

### 2.3 Robustness binary — Anomalous vs. not

`y = 1` if `|residual_std| > 1.0`, else `y = 0`. Captures over- and
under-priced systems symmetrically; most useful for operational anomaly
monitoring and for comparison with [notebooks/Anomaly Detection-1.ipynb](../../notebooks/Anomaly%20Detection-1.ipynb)'s
`|z| > 2.5` threshold.

## 3. Leakage safeguards specific to residual-derived labels

Residual-based labels can leak price into the classifier if built carelessly.
The refactored regression notebook already does the right thing, but the
safeguards are repeated here so the classification notebook can be audited
against them:

1. The regression model is trained on the train split only.
2. Train residuals come from `cross_val_predict` with 5-fold CV, i.e., each
   row's residual is computed from a model that did **not** see that row
   during fitting.
3. Test residuals come from a single held-out fit on all train data.
4. Quantile cut-points for class boundaries are fit on the train residual
   distribution only; they are then applied to test residuals without
   re-fitting.
5. The classifier receives the **spec features** (the same ones used by the
   regression preprocessor), **not** the residual and **not** any function of
   price. Feeding the residual into the classifier would be circular.

## 4. Imbalance handling

With the 10 / 40 / 40 / 10 cut-points, both "Steal" and "Extreme Tax"
classes are minority. The notebook addresses this in three ways:

1. **Stratified splits.** The Part 3 notebook uses stratified 5-fold CV on
   the multiclass label; the refactored regression's own 80/20 split is
   stratified on `device_type` and is re-used here.
2. **Class weighting.** Every sklearn model that supports it is trained with
   `class_weight="balanced"` (or `sample_weight` derived from inverse class
   frequency for GBM).
3. **Optional SMOTE.** If `imblearn` is available, a sensitivity check runs
   the winning model with SMOTE on the training fold only. If `imblearn` is
   missing we skip this and document it.

The empirical class distribution on train and test is printed at the top of
the classification notebook so the professor's imbalance question can be
answered with actual numbers.

## 5. Model slate

All models share the preprocessor defined in the refactored regression
notebook (tiered encoding with RobustScaler, one-hot for low-cardinality,
target encoding for cpu_model and gpu_model, frequency encoding for brand,
`model` dropped as an identifier). The pipeline is constructed once and
cloned for each estimator so we can be confident the comparison is fair.

| role                 | estimator                                                       |
|----------------------|-----------------------------------------------------------------|
| interpretable base   | `LogisticRegression(multi_class='multinomial')`                 |
| linear margin        | `LinearSVC(class_weight='balanced')`                            |
| non-linear kernel    | `SVC(kernel='rbf', probability=True)` on a 20k stratified subsample |
| non-linear rules     | `DecisionTreeClassifier(class_weight='balanced')`               |
| strong non-linear    | `HistGradientBoostingClassifier` (fast GBM at this scale)       |
| ensemble (Prof ask)  | `VotingClassifier(voting='soft')` over Logistic + SVC-RBF + HistGBM + DT |

Tuning uses `RandomizedSearchCV` with a small, documented grid per model so
the runtime budget stays under a few minutes on a laptop. Two stretch
options are listed but are not required: a `StackingClassifier` with a
logistic meta-learner, and `XGBClassifier`.

## 6. Evaluation protocol

- **Train / test:** re-use the exact 80/20 `random_state=1234` split from
  the regression notebook so residuals line up row-for-row.
- **CV inside train:** stratified 5-fold for hyperparameter search.
- **Headline metrics (multiclass):**
  - Macro F1 (primary — imbalance-aware)
  - Balanced accuracy
  - Per-class precision / recall / F1
  - Confusion matrix on test
  - Multiclass ROC-AUC one-vs-rest (macro)
  - Multiclass PR-AUC (macro)
  - Log-loss
  - Brier score per class
  - Calibration curve (especially important before soft-voting)
- **Headline metrics (binary variants):** ROC-AUC, PR-AUC, F1, balanced
  accuracy, confusion matrix at 0.5 and at the F1-optimal threshold.
- **Interpretation:** odds-ratio table from multinomial Logistic, decision
  tree top levels visualized, permutation importance on the winning
  ensemble.
- **Error analysis:** pull 10 examples each of the most common confusions
  (e.g. Brand Premium predicted as Extreme Tax) and discuss in the report.

## 7. Recommendation

- **Headline classifier for the report:** the soft-voting ensemble on the
  multiclass Value Score label.
- **Interpretability companion:** multinomial Logistic with odds-ratio table.
- **Robustness checks (short sub-sections):** Steal-vs-not and Anomalous-vs-not
  binary models.

Classification should be **in scope** in the final report, not optional.
The residual-based multiclass framing operationalizes the project's
"Brand Tax" thesis in a way that the regression-only track cannot.

## 8. What's left (roadmap across the whole project)

Ties the remaining pieces back to the README workflow (steps 4-8) so the team
has a single checklist:

| Step                                      | Deliverable                                                                                                   | Status  |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------|---------|
| 4. Baseline linear regression             | [notebooks/Project_refactored.ipynb](../../notebooks/Project_refactored.ipynb) Part 3 (OLS, Ridge, Lasso) | done    |
| 5. Feature engineering and encoding       | Part 2 tiered preprocessor; Part 1 predictor audit                                                            | done    |
| Prof Q1 — predictor count audit            | Part 1 audit table                                                                                             | done    |
| Prof Q2 — numeric as categorical          | Part 7 binning experiment                                                                                     | done    |
| Prof Q3 — imputation comparison           | Part 6 imputation experiment                                                                                  | done    |
| 6. Residual analysis                      | Part 5 and `reports/residuals.parquet`                                                                         | done    |
| 7. Feature importance                     | Part 4 permutation importance + VIF                                                                            | done    |
| Classification track (multiclass primary) | [notebooks/Classification.ipynb](../../notebooks/Classification.ipynb) and this document                      | in progress |
| Ensemble model                            | Classification notebook Voting section                                                                         | in progress |
| 8. Final report and presentation          | `reports/outline.md` scaffold (populated by the team)                                                          | pending |
| Cleanup                                   | Archive `docs/data-notes/Project.ipynb` (superseded by refactored) and `notebooks/Anomaly Detection-1.ipynb` (absorbed into Part 5 + binary robustness check) | pending |
