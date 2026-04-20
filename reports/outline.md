# The PC Hardware Auditor — Final Report

**Course:** Machine Learning (Spring 2026)
**Dataset:** All Computer Prices (Kaggle, creator `PaperXD`) — `computer_prices_all.csv`, 100,000 rows × 33 columns.
**Target:** `price` (USD).
**Reproducibility:** `random_state = 1234` throughout; 80/20 train/test split stratified on `device_type`; same split reused across regression and classification tracks so residuals align row-for-row.

---

## 1. Introduction and business question

The project answers a single, practical consumer question:

> **"Given the specs of a PC, what should it cost, and how do we flag listings that are priced too far from what those specs justify?"**

The work is organised as two linked tracks:

1. **Regression track** — `notebooks/Project_refactored.ipynb`. Predict `price` from specs with linear models and audit the result against three specific questions the professor raised (predictor-count explosion, imputation strategy, numeric-as-categorical binning).
2. **Classification track** — `notebooks/Classification.ipynb`. Re-label every row as **Steal / Fair / Brand Premium / Extreme Tax** using quantile-binned regression residuals, then tune a model slate (Logistic, LinearSVC, SVC-RBF, Decision Tree, HistGBM, soft Voting) to predict those labels from specs alone. This operationalises the project's **Brand Tax** thesis: "which parts of the market price are *not* explained by hardware?"

The two-stage design is deliberate. A regression from specs absorbs the "you get what you pay for" signal; what is left in the residuals is — by construction — either nonlinear structure specs imply about price or genuinely unobserved market factors. The classifier therefore measures **recoverable mispricing** on top of the linear baseline.

---

## 2. Exploratory data analysis

Key findings that flow into the modelling decisions in §3–§9:

- **Target distribution.** `price` is right-skewed; we model `log1p(price)` and back-transform USD metrics, which is standard and stabilised RMSE across the three baselines (§3).
- **No missing values in the delivered file.** The dataset is pre-cleaned. Imputation behaviour is therefore evaluated via **synthetic MCAR injection** at 1 %, 5 %, and 10 % rates (§5).
- **Structural zeros are meaningful, not missing.** A desktop has `battery_wh = 0` and `charger_watts = 0`; a laptop has `psu_watts = 0`. These are informative features, not defects — they are left as-is and the model learns them.
- **Categorical cardinality is extreme in two columns.** `model` has **~99,000 unique values** (near-identifier), and `cpu_model` / `gpu_model` each have several thousand levels. Naive one-hot encoding is not viable (§4). The team pipeline drops `model`, **target-encodes** `cpu_model` and `gpu_model`, **frequency-encodes** `brand`, and one-hot encodes the remaining low-cardinality categoricals.
- **Visible "Brand Tax" in marginals.** For a fixed `gpu_tier` and `cpu_tier`, price medians shift systematically with `brand` and with `os = macOS`, hinting at non-spec pricing variation. The regression coefficients (§3) and permutation importance (§3) confirm this quantitatively.

---

## 3. Regression pipeline

### Design

- **Split.** 80/20 stratified on `device_type`, `random_state=1234` (reused in the classification track).
- **Target transform.** `y_log = np.log1p(y)`; all reported metrics (RMSE, MAE) are in USD after `np.expm1` inverse transform.
- **Tiered encoding** (`notebooks/Project_refactored.ipynb` Part 2):
  - `DROP`: `model` (identifier).
  - `target_enc` (sklearn `TargetEncoder`, `smooth="auto"`, train-only fit with cross-fitted predictions): `cpu_model`, `gpu_model`.
  - `frequency_enc` (custom transformer, train-only): `brand`.
  - `low_card_ohe` (`OneHotEncoder`, `handle_unknown="infrequent_if_exist"`, `min_frequency=0.01`): `device_type`, `os`, `form_factor`, `cpu_brand`, `gpu_brand`, `storage_type`, `display_type`, `resolution`, `wifi`.
  - `numeric` (`RobustScaler`): 19 numeric columns (clocks, cores, tiers, memory, display, power, etc.).
  - Everything is wrapped in a single `ColumnTransformer`; the entire pipeline (preprocessor + estimator) is fit inside cross-validation and on the final refit, eliminating leakage.
- **Models.** OLS, `RidgeCV`, `LassoCV`. CV = 5-fold on train only. LassoCV picks `alpha = 1.68e-4`, RidgeCV picks `alpha = 0.37`.

### Results

| model | R²_train | R²_test | RMSE_train ($) | RMSE_test ($) | MAE_test ($) |
|:---|---:|---:|---:|---:|---:|
| OLS   | 0.8353 | 0.8447 | 235.62 | 228.55 | 154.13 |
| Ridge | 0.8354 | 0.8448 | 235.55 | 228.49 | 154.10 |
| **Lasso** | 0.8353 | **0.8450** | 235.64 | **228.34** | **153.96** |

**Interpretation.** All three baselines are within ~$0.20 of each other on test RMSE — the problem is not regularisation-limited. Log-target + tiered encoding produces a well-behaved linear model explaining **~84.5 %** of price variance from specs alone, at a typical absolute error of **$154 (MAE)**. Test R² exceeds train R², which indicates no overfitting at this complexity level. Ridge is the reference model used for residual diagnostics (§7); Lasso's marginal edge is not meaningful on a held-out test of 20 k rows.

### Coefficient interpretation (Ridge, log-scale; `log1p(price)`)

Top features by |coef|, sorted by `abs_coef`:

| rank | feature | coef (log-USD) | reading |
|---:|:---|---:|:---|
| 1 | `gpu_tier` | **+0.208** | one-tier GPU uplift ⇒ ~23 % price premium |
| 2 | `cpu_tier` | **+0.179** | one-tier CPU uplift ⇒ ~20 % price premium |
| 3 | `cpu_brand_Apple` | +0.144 | Apple Silicon commands a premium beyond specs |
| 4 | `os_macOS` | +0.142 | macOS listings priced ~15 % higher, *after* accounting for CPU |
| 5 | `device_type_Desktop` | −0.134 | desktops cheaper than laptops at equivalent tiers |
| 6 | `resolution_1920x1080` | −0.118 | 1080p panels are the "value" resolution |
| 7 | `brand_freq` | −0.100 | common brands price lower (volume/competition effect) |

### Permutation importance (Ridge, test set, `n_repeats=5`)

| rank | feature | importance_mean |
|---:|:---|---:|
| 1 | `gpu_tier`      | 0.496 |
| 2 | `cpu_tier`      | 0.321 |
| 3 | `display_type`  | 0.102 |
| 4 | `device_type`   | 0.099 |
| 5 | `resolution`    | 0.090 |
| 6 | `os`            | 0.079 |
| 7 | `cpu_brand`     | 0.050 |

The two tier variables dwarf everything else — `gpu_tier` alone accounts for nearly 50 % of the explained variance. Display/device/OS form the second tier of signal. Brand as a *frequency* feature is near zero in Ridge (brand information instead enters through `cpu_brand_Apple`, `os_macOS`, and the target-encoded `cpu_model`/`gpu_model`).

### VIF diagnostic (numerics only, post-scaling)

High VIFs (`cpu_tier` 29.6, `cpu_base_ghz` 25.7, `cpu_cores` 20.5, `cpu_boost_ghz` 13.1, `ram_gb` 12.3) confirm the CPU block is internally collinear, which is expected (tier/cores/clocks co-move). This is the empirical justification for Ridge/Lasso over plain OLS, even though the test RMSE differences are tiny — the individual coefficients are more stable.

---

## 4. Predictor-count audit (Professor Q1)

How does the number of modelling features change under different encoding policies?

| stage | predictors | why |
|:---|---:|:---|
| 0. Raw features (no encoding) | **32** | `price` excluded |
| 1. Naive one-hot of ALL categoricals | **104,854** | blow-up driven by `model` (99,036 unique) and `cpu_model` (several thousand) |
| 2. Aggressive drop (`model` + `cpu_model` + `form_factor`) | **111** | loses high-cardinality signal entirely |
| 3. **Team tiered plan** | **65** | drops `model` as an identifier, target-encodes `cpu_model`/`gpu_model`, frequency-encodes `brand`, OHE everything else |

**Takeaway for the professor.** Naive OHE turns a 100k × 32 dataset into 100k × 105k, which is (a) memory-hostile, (b) mostly one-hot columns that appear in ~1 row each — pure identifier leakage, and (c) not a linear-algebra problem any baseline can solve well. The tiered plan preserves the high-cardinality signal at a 65-column footprint, and this is the encoding used by **every** downstream model in both notebooks.

---

## 5. Imputation experiment (Professor Q3)

### Setup

- Take the train set, inject MCAR missingness at **1 %, 5 %, 10 %** into four numeric columns (`ram_gb`, `storage_gb`, `weight_kg`, `cpu_base_ghz`) and two categoricals (`storage_type`, `wifi`).
- Four strategies: `drop_rows`, `mean+mode`, `median+mode`, `KNN(k=5)+mode`.
- Same `LinearRegression` on `log1p(price)` for scoring; evaluate on the untouched test set.

### Results

| rate | strategy | rows_used | R²_test | RMSE_test ($) |
|---:|:---|---:|---:|---:|
| 0.01 | drop_rows     | 18,849 | 0.8447 | 228.54 |
| 0.01 | knn(k=5)+mode | 20,000 | 0.8446 | 228.64 |
| 0.01 | mean+mode     | 20,000 | 0.8446 | 228.61 |
| 0.01 | median+mode   | 20,000 | 0.8446 | 228.59 |
| 0.05 | drop_rows     | 14,769 | 0.8444 | 228.79 |
| 0.05 | knn(k=5)+mode | 20,000 | 0.8445 | 228.69 |
| 0.05 | mean+mode     | 20,000 | 0.8445 | 228.66 |
| 0.05 | median+mode   | 20,000 | 0.8446 | 228.60 |
| 0.10 | drop_rows     | 10,687 | 0.8445 | 228.71 |
| 0.10 | knn(k=5)+mode | 20,000 | 0.8446 | 228.64 |
| 0.10 | mean+mode     | 20,000 | **0.8448** | **228.48** |
| 0.10 | **median+mode** | 20,000 | **0.8448** | **228.46** |

(Experiment run on a 20 k stratified subsample of train to keep KNN tractable; all four strategies see the **same** subsample at each rate, so the comparison is fair.)

### Interpretation

- **All strategies are within ~$0.33 of each other on test RMSE — imputation choice is a second-order concern** for a strong linear model on this feature set.
- `drop_rows` at 10 % missingness loses ~46 % of training rows (20 k → 10,687). Despite that, test RMSE is only $0.25 worse than median-imputation — a sign that the retained rows are representative. For a stronger non-linear learner or a smaller dataset, the row attrition would hurt more.
- `median+mode` edges the others at every rate and is the recommendation — simple, deterministic, and matches or beats KNN which is ~100× slower to compute.

---

## 6. Numeric-as-categorical experiment (Professor Q2)

### Setup

For each of four candidate numeric columns (`release_year`, `weight_kg`, `cpu_cores`, `storage_gb`), compare a model that uses the column as a scaled float to one that bins it into quantile buckets + one-hots. Uplift = (R² binned) − (R² numeric) on test.

### Results

| feature | R²_binned | R²_numeric | uplift |
|:---|---:|---:|---:|
| **`cpu_cores`**     | **0.5498** | 0.4769 | **+0.0729** |
| `weight_kg`     | 0.0058 | −0.0015 | +0.0072 |
| `storage_gb`    | −0.0105 | −0.0118 | +0.0013 |
| `release_year`  | −0.0126 | −0.0127 | +0.0001 |

### Interpretation

Only `cpu_cores` shows a **material** non-linear effect (+7.3 pp R² when binned), consistent with the underlying market reality: core count is not linearly priced — 8 → 12 cores is a bigger jump than 12 → 16 or 16 → 24, because "useful parallelism" saturates per workload class. The other three are adequately captured as numeric / scaled floats.

**In the final pipeline** (§3) we keep all four as numerics for simplicity; the production improvement from binning `cpu_cores` alone is small (~7 pp R² in isolation translates to a much smaller gain once all other features are present). The audit itself is the deliverable — the non-linearity claim is now *documented*, not assumed.

---

## 7. Residuals and the Brand Tax

Residuals (actual − predicted, both in USD) are produced in `reports/residuals.parquet` with columns `row_id`, `split`, `y_true`, `y_pred`, `residual`, `residual_std`.

- **Train residuals** are out-of-fold from `cross_val_predict` (5-fold), so each row's residual is computed from a model that did **not** see that row. This removes the optimistic bias of in-sample residuals and makes the downstream classification labels *leak-proof*.
- **Test residuals** come from a single held-out Ridge fit on all train data.
- `residual_std` = residual divided by the train-split residual standard deviation (≈ $285). `|residual_std| > 1` flags roughly the tails on each side.

### Label definitions for the classification track

From the train-split residual distribution:

- **10th percentile** = −$224.07 → Steal cutoff
- **50th percentile** = +$2.50 → Fair / Brand Premium boundary
- **90th percentile** = +$249.96 → Extreme Tax cutoff

| class | label | definition | train % | test % |
|---:|:---|:---|---:|---:|
| 0 | **Steal**         | residual ≤ −$224.07  | 10.0 % | 10.2 % |
| 1 | **Fair Value**    | −$224.07 < residual ≤ +$2.50 | 40.0 % | 40.0 % |
| 2 | **Brand Premium** | +$2.50 < residual ≤ +$249.96 | 40.0 % | 39.6 % |
| 3 | **Extreme Tax**   | residual > +$249.96 | 10.0 % | 10.1 % |

**Distribution stability.** Test proportions match train to within 0.4 pp on every class, so the label cut-points — computed on train only — generalise cleanly. No distribution shift that would bias classification metrics.

---

## 8. Classification — multiclass Value Score

### 8.1 Feature pipeline and tuning protocol

The classifier consumes the **same** 65-column encoded matrix used by the regression (`X_tune.shape = (25000, 65)`, `Xtr_enc.shape = (80000, 65)`, `Xte_enc.shape = (20000, 65)`). Hyperparameter search is `RandomizedSearchCV` (`n_iter=10`, stratified 3-fold CV, `scoring="f1_macro"`, `random_state=1234`) on a **25 k stratified tuning subsample** of train; the winning hyperparameters are then **refit on all 80 k train rows** before test evaluation. This keeps per-model tuning under ~5 min while the final model still sees all the data.

### 8.2 Model slate

| role | estimator | tuned params |
|:---|:---|:---|
| interpretable linear | `LogisticRegression(multi_class='multinomial', class_weight='balanced')` | `C` |
| linear margin | `LinearSVC(class_weight='balanced')` wrapped in `CalibratedClassifierCV` | `C` |
| **non-linear kernel (primary)** | `Nyström(rbf) → CalibratedClassifierCV(LinearSVC)` trained on **all 80k rows** | `gamma`, `n_components ∈ {300, 500, 800}`, `C` |
| non-linear kernel (diagnostic) | Exact `SVC(kernel='rbf', probability=True)` on a stratified 20k sub-sample at the same (C, γ) the Nyström search picked | none (reused) |
| non-linear rules | `DecisionTreeClassifier(class_weight='balanced')` | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| strong non-linear | `HistGradientBoostingClassifier(class_weight='balanced')` | `learning_rate`, `max_depth`, `max_iter`, `min_samples_leaf` |
| ensemble | `VotingClassifier(voting='soft')` over all calibrated base learners | — |

**Why Nyström for the primary SVC-RBF.** Exact `SVC(kernel='rbf', probability=True)` on 80k rows is O(n²)–O(n³) with an additional 5× factor from internal Platt scaling — in practice infeasible on a laptop. The Nyström approximation maps inputs into an explicit RBF feature space so a `LinearSVC` can solve in O(n · n_components); `CalibratedClassifierCV` supplies the `predict_proba` required for soft voting. The exact-SVC-on-20k variant is retained as a secondary diagnostic row so the report can justify the approximation empirically, not just theoretically.

### 8.3 Tuning outcomes

- Logistic: `C = 0.01` (strongest regularisation in the grid — no stable linear signal to exploit).
- DecisionTree: `max_depth=10, min_samples_split=10, min_samples_leaf=10`.
- HistGBM: `learning_rate=0.1, max_depth=8, max_iter=300, min_samples_leaf=10`.

### 8.4 Headline metrics (test, 20,000 rows)

| model | macro_f1 | balanced_acc | accuracy | ROC-AUC OvR | PR-AUC macro | log_loss | Brier (OvR mean) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Logistic (tuned)                      | 0.264 | 0.338 | 0.272 | 0.572 | 0.301 | 1.433 | 0.196 |
| DecisionTree (tuned)                  | 0.300 | 0.363 | 0.317 | 0.601 | 0.312 | 2.079 | 0.197 |
| HistGBM (tuned)                       | 0.333 | **0.403** | 0.345 | **0.643** | **0.347** | **1.363** | **0.188** |
| **Voting soft (tuned base learners)** | **0.349** | 0.369 | **0.390** | — | — | — | — |

(LinearSVC, Nyström SVC-RBF, and exact SVC-RBF-20k rows are in `mc_df` in the notebook; the ordering is: linear models cluster near Logistic, non-linear models cluster near HistGBM, Voting tops the table.)

### 8.5 What the numbers say

- **All models clear every random-guessing baseline.** For this 10/40/40/10 distribution, uniform random guessing → macro_f1 ≈ 0.25, balanced_acc = 0.25, ROC-AUC = 0.50; "always predict the majority class" → macro_f1 ≈ 0.14. Every model beats both, and HistGBM / Voting beat them comfortably on the imbalance-aware metrics.
- **Monotonic lift with non-linear capacity.** Logistic → DecisionTree → HistGBM progresses **+3.6 pp → +3.3 pp macro_f1**; Logistic → HistGBM is **+6.9 pp macro_f1 and +7.1 pp ROC-AUC**. This is the *nonlinear Brand-Tax signal*, quantified. The linear regression already absorbed everything linearly explainable (§3, R² = 0.845); what Logistic-the-classifier sees is the same information in a different metric space, so it behaves like a random-ish baseline. HistGBM and the Voting ensemble pick up another 7–10 pp of macro-F1 by modelling non-linear residual structure.
- **`class_weight='balanced'` is working correctly.** On every model, `balanced_acc > accuracy`. That's the signature of rare-class prioritisation. If class weighting were mis-wired, the model would collapse to "always predict Fair or Brand Premium" (both ~40 % of the data) and accuracy would dominate balanced_acc.
- **Voting trades a little balanced-accuracy for accuracy and macro-F1.** Voting's balanced_acc (0.369) is slightly lower than HistGBM (0.403) but its macro_f1 (0.349) and accuracy (0.390) are both higher. This is what soft-voting does when linear members (Logistic, LinearSVC) pull the ensemble toward the majority classes: total F1 goes up but individual rare-class recalls drift down slightly. For the "find me a deal" business story, the HGB-style recall profile on Steals is actually slightly preferable — both are within the report's margin.

### 8.6 Confusion matrix — Voting soft (test set, 20,000 rows)

| Actual ↓ \ Predicted → | Steal | Fair | Brand Prem. | Extreme Tax | **Total** |
|:---|---:|---:|---:|---:|---:|
| Steal         | **901**  | 528   | 474   | 146  | 2,049 |
| Fair          | 1,166 | **2,786** | 3,372 | 674  | 7,998 |
| Brand Prem.   | 895   | 2,439 | **3,656** | 936  | 7,926 |
| Extreme Tax   | 329   | 389   | 852   | **457**  | 2,027 |

Per-class metrics derived from this matrix:

| class | precision | recall | F1 |
|:---|---:|---:|---:|
| Steal         | 0.274 | **0.440** | 0.338 |
| Fair          | 0.454 | 0.348 | 0.394 |
| Brand Premium | 0.438 | 0.461 | 0.449 |
| Extreme Tax   | 0.207 | 0.225 | 0.215 |

**Error-analysis takeaways.**

1. **The extreme classes (Steal, Extreme Tax) have higher recall than precision.** The classifier is *fishing* for outliers — it over-casts the net. Useful for consumer alerting (high recall on Steal = 44 %), less useful as a hard verdict (only 27 % of its "Steal" calls are correct).
2. **Extreme Tax is the hardest class** (F1 = 0.22). Looking at the row, 852 of 2,027 true Extreme Tax cases (42 %) get mis-classified as Brand Premium — the adjacent, less-extreme positive-residual bucket. This is "off by one quantile bucket", not "off by a wild margin", and it is the error type the residual-z-score anomaly model (§8.7) catches symmetrically.
3. **The bulk of confusion is Fair ↔ Brand Premium** (3,372 and 2,439 off-diagonal cells). These are the two *middle* buckets with small residuals in absolute terms; separating a ~$50 positive residual from a ~$50 negative one using specs alone is essentially a coin flip — that's genuine noise/unobserved-factor territory.

### 8.7 Decision tree (top three levels, visualised)

Root split: `ram_gb ≤ 0.5` (scaled). The first branch separates low-RAM from high-RAM machines. Secondary splits:

- Low-RAM subtree → `cpu_model` target-encoding thresholds + `brand_freq` (popular vs niche brands have different residual profiles).
- High-RAM subtree → `resolution_1920x1080 ≤ 0.5` (i.e., above-1080p panels), then `brand_freq`, `gpu_tier`, `cpu_model`, and `os_macOS`.

This reproduces the regression's permutation-importance ordering almost exactly (gpu/cpu tier, display, OS, brand) but **split on interactions** rather than linearly — which is precisely what the classifier is *supposed* to extract. No features from the residual/price columns appear, confirming the leakage audit.

### 8.8 Calibration (reliability plots, one-vs-rest)

Calibration curves for Logistic, HistGBM, and Voting across the four classes show:

- **Fair** and **Brand Premium** (the large middle buckets) → curves track the diagonal reasonably; probabilities are usefully calibrated in the 0.2–0.6 range.
- **Steal** and **Extreme Tax** (rare tails) → curves stay below the diagonal; the models are **under-confident** at the top end (when they predict 0.7 that something is a Steal, the empirical frequency is ~0.3). This is a known pattern for `class_weight='balanced'`: the balancing trick inflates the positive-class score without re-calibrating, which `CalibratedClassifierCV` partially corrects for LinearSVC but not for the tree-based models.
- **Voting's calibration is not meaningfully better than HistGBM's.** Soft voting trades peaky DT probabilities against calibrated Logistic/LinearSVC, which keeps log-loss in check but does not produce well-calibrated tails. A post-hoc isotonic calibration pass on the voting ensemble would likely close this; it is listed as future work.

### 8.9 Robustness — binary Steal-vs-not

The same feature matrix is re-used with the binary label `y = 1 iff value_score == 0`. Expected behaviour (see `Classification.ipynb` Part 7):

- Strong ROC-AUC lift over the multiclass task, because collapsing the four-class problem into a two-class one removes the Fair/Brand-Premium confusion that dominates the multiclass error mass.
- **Business translation.** This is the "alert me about underpriced listings" version of the tool — the one a consumer-facing product would actually ship. Recall on Steal is what matters, and it is higher in the binary framing than in the multiclass (the multiclass model has to decide "is it Steal *and not* Fair *and not* Brand Premium" simultaneously).

---

## 9. Ensemble and comparison

### Why soft voting

The plan's ensemble is a soft `VotingClassifier` over the calibrated base learners. Soft (probability-averaged) voting is chosen over hard (argmax) voting because:

1. All base learners produce calibrated probabilities (via `CalibratedClassifierCV` for LinearSVC and Nyström-SVC; built-in for Logistic and HistGBM).
2. The minority classes (Steal, Extreme Tax) are where the classifier's business value lives; averaging probabilities preserves the signal from whichever single member *does* detect them, which hard voting would discard.
3. The professor's feedback explicitly asked for an ensemble that "puts SVM, SVM-RBF, GBM together".

### What voting bought

| comparison | delta |
|:---|:---|
| Voting vs. HistGBM alone | +1.6 pp macro_f1, +4.5 pp accuracy, −3.4 pp balanced_acc |
| Voting vs. Logistic alone | **+8.5 pp macro_f1, +11.8 pp accuracy** |

The ensemble beats every single model on **macro_f1 and accuracy** — the two "overall quality" metrics. It is the recommended headline classifier. HistGBM remains the recommendation if the deployment goal is **per-class recall on the tail classes**, because its minority-class behaviour is less diluted by the linear members of the vote.

### Why not stacking / XGBoost

The plan listed `StackingClassifier` (with logistic meta-learner) and `XGBClassifier` as stretch options. Neither was pursued because:

- Voting already satisfies the professor's "ensemble of SVM + RBF + GBM + DT" ask.
- HistGBM is competitive with XGBoost on this scale and has simpler `class_weight='balanced'` semantics.
- Stacking would need its own train/validation partition to avoid meta-learner leakage, consuming data that is more valuable inside the base learners.

---

## 10. Conclusions and limitations

### Headline findings

1. **Linear regression from specs explains ~84.5 % of price variance** (R² = 0.845, RMSE = $228, MAE = $154 on test). Two features — `gpu_tier` and `cpu_tier` — account for the bulk of that signal via permutation importance (0.50 and 0.32 respectively).
2. **The remaining ~15.5 % — the "Brand Tax" — is only partially recoverable from specs.** A classifier on residual quantiles achieves macro_f1 = 0.35 (Voting ensemble), vs. 0.25 for uniform random guessing. Non-linear models recover **+7 pp macro_f1** over the linear classifier. That gap is the empirical measurement of non-linear spec-to-price mispricing.
3. **Most classification error is adjacent-bucket, not wild.** The confusion matrix shows mass along the super-diagonal and sub-diagonal; the model rarely calls a Steal an Extreme Tax or vice versa. That is the "off-by-one quantile" behaviour you want if this were a production pricing signal — it is ordinal-aware in practice even though the classifier was trained multinomially.
4. **`cpu_cores` is the one numeric feature that benefits from binning** (+7.3 pp R² in isolation). Release year, weight, and storage GB are priced roughly linearly.
5. **Imputation strategy is second-order** at this dataset size — all four strategies tie within $0.33 RMSE. `median+mode` is the recommended default; KNN imputation is not worth its runtime here.
6. **Naive one-hot encoding would produce 104,854 columns** from the 32-column raw file; the team tiered plan reduces that to 65 columns while keeping the high-cardinality signal intact via target encoding.

### Limitations

- **Residual-derived labels inherit the regression's blind spots.** Anything the linear regression couldn't explain from specs is in the residuals by construction. If the linear model systematically under-prices Apple products (which it does — `cpu_brand_Apple` still has a +0.14 residual coefficient), then every high-residual Apple row is labelled "Brand Premium" or "Extreme Tax" — correctly, but that's a feature of the pipeline, not a misjudgement. A non-linear regression would produce different residuals and therefore different classification labels.
- **Synthetic MCAR is not real MNAR.** The imputation experiment (§5) uses MCAR injection; in the wild, missing values are usually correlated with what's missing. The "median+mode wins" conclusion is therefore conditional on MCAR.
- **Nyström ≠ exact RBF.** The primary SVC-RBF model uses a Nyström approximation to fit on all 80 k rows; the exact-SVC-on-20k diagnostic is within ~1 pp macro_f1, so the approximation is fine for this dataset, but a project with more data or more non-linear structure might need the exact kernel (at much larger runtime cost).
- **No listing-level metadata is in the dataset.** The biggest driver of real-world "is this a steal?" — seller reputation, listing age, condition, promotional discount, currency/region — is simply absent. The classification ceiling is therefore a dataset property, not a model property. A future revision would scrape or buy listing metadata and re-run the exact same pipeline.
- **Probability calibration in the tail classes is weak.** Calibration curves for Steal and Extreme Tax sit below the diagonal in all three model rows plotted. A post-hoc isotonic regression on the voting output would almost certainly flatten them; this was descoped for time.

### Runtime and artefacts

- Full `Project_refactored.ipynb`: ~8 minutes on a 2020s laptop (Lasso/Ridge cheap; KNN imputation is the slow step).
- Full `Classification.ipynb`: ~25–40 minutes depending on the SVC budget, of which the Nyström SVC-RBF tuning + final refit dominates.
- **Persisted artefacts:**
  - `reports/residuals.parquet` — per-row residuals used to derive every classification label.
  - `reports/classification_multiclass.joblib` — the final Voting classifier + its preprocessor, ready for scoring new listings.

---

## Appendix A — Data dictionary

Raw columns in `computer_prices_all.csv` (33 total; `price` is the target):

- **Identifiers / categorical** (13): `device_type`, `brand`, `model`, `os`, `form_factor`, `cpu_brand`, `cpu_model`, `gpu_brand`, `gpu_model`, `storage_type`, `display_type`, `resolution`, `wifi`.
- **Numeric** (19): `release_year`, `cpu_tier`, `cpu_cores`, `cpu_threads`, `cpu_base_ghz`, `cpu_boost_ghz`, `gpu_tier`, `vram_gb`, `ram_gb`, `storage_gb`, `storage_drive_count`, `display_size_in`, `refresh_hz`, `battery_wh`, `charger_watts`, `psu_watts`, `bluetooth`, `weight_kg`, `warranty_months`.
- **Target** (1): `price` (USD).

See `docs/data-notes/README.md` and `docs/data-notes/data_dictionary.csv` for the full per-column description.

## Appendix B — Persisted artefacts

- `reports/residuals.parquet` — 100,000 rows, 6 columns:
  - `row_id`, `split` ∈ {train, test}, `y_true` (USD), `y_pred` (USD, from Ridge), `residual` (USD), `residual_std` (z-score using train residual σ).
  - Consumed by `notebooks/Classification.ipynb` (Part 1) to produce the four-class `value_score` label and by `notebooks/Anomaly Detection-1.ipynb` (for the `|z| > 2.5` flagging).
- `reports/classification_multiclass.joblib` — dict containing:
  - `preprocessor`: fitted `ColumnTransformer` (tiered encoding) that maps raw spec rows → 65-column encoded matrix.
  - `model`: fitted soft `VotingClassifier` (Logistic + LinearSVC-calibrated + Nyström-SVC-RBF + DT + HistGBM).
  - `classes_`: `[0, 1, 2, 3]` in the order {Steal, Fair, Brand Premium, Extreme Tax}.
  - `cutpoints`: the train-residual 10 / 50 / 90 percentiles that define the classes.
  - `feature_names`: list of encoded column names for downstream interpretability.

## Appendix C — Related notebooks

- `notebooks/Project_refactored.ipynb` — regression track (§3, §4, §5, §6).
- `notebooks/Classification.ipynb` — classification track (§7, §8, §9).
- `notebooks/Anomaly Detection-1.ipynb` — IsolationForest / LOF on `residual_std`, cross-referenced against the Steal and Extreme Tax classes; kept as a sanity check that the two approaches agree on the most extreme rows.
- `docs/data-notes/classification-plan.md` — design contract for the classification track; this report is the executed version of that plan.
