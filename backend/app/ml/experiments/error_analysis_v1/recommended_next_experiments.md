# Recommended Next Experiments: Resolving the Subcategory Accuracy-Recall Trade-Off

## Objective
Recover the 5.59% accuracy lost in `dataset_reduction_v1` while preserving the minority-class recall and macro-F1 improvements achieved through class rebalancing.

## Recommended Next Experiments (Ranked by Evidence & Expected Impact)

### 🥇 Experiment 1: Soft-Cap (Square-Root / Log-Frequency) Stratification (Recommended First)
- **Diagnostic Basis**: The hard 600-sample cap over-compressed `Garbage Dumping` (3,465 $\to$ 600, an 82.7% reduction) and `Potholes` (2,228 $\to$ 600, a 73.1% reduction), bringing them too close to small adjacent classes (e.g., Street Cleaning with 600 samples). This directly caused 135 regressions.
- **Proposed Intervention**: Implement a **damped soft cap** using square-root or log frequency:
  $$N_{\text{train}}(c) = \min\left(N_{\text{raw}}(c), N_{\text{base}} + k \cdot \sqrt{N_{\text{raw}}(c)}\right)$$
  For example, cap majority classes at **1,000–1,200 samples** instead of 600, while oversampling or preserving minority classes at 100–300.
- **Hypothesis**: Retains sufficient majority prior to preserve the ~78 `Garbage Dumping` and ~38 `Potholes` true positives, while maintaining enough minority representation to preserve the F1 gains in `Waste Disposal` and `Street Light Installation`.
- **Expected Metric Impact**: +3.5% to +4.5% E2E accuracy recovery with Macro-F1 $\ge 0.52$.

### 🥈 Experiment 2: Intra-Branch Threshold Tuning & Asymmetric Prior Calibration
- **Diagnostic Basis**: Within `Waste & Sanitation`, `Garbage Dumping` and `Street Cleaning` have overlapping vocabulary. CalibratedClassifierCV outputs probabilities, but the argmax decision rule treats all class boundaries at equal 0.5 probability.
- **Proposed Intervention**: Introduce a **class-specific decision threshold vector** $\boldsymbol{\tau} = \{\tau_1, \dots, \tau_K\}$ tuned on the validation set to maximize Macro-F1 subject to an accuracy floor constraint ($\text{Acc} \ge 70.0\%$).
- **Implementation**: Zero retraining required; can be applied immediately to existing `task3_subcategory_classifier_bundle_v1.joblib` via validation sweep.

### 🥉 Experiment 3: Distinguishing Lexical Features for Collocated Subcategories
- **Diagnostic Basis**: 28 regressions were `Garbage Dumping` misclassified as `Street / Public Area Cleaning`. Both classes share words like *'garbage'*, *'waste'*, *'street'*, *'road'*. Unigrams and general bigrams blur this boundary.
- **Proposed Intervention**: In the `Waste & Sanitation` branch pipeline, add specialized collocations (e.g. *'sweeper'*, *'broom'*, *'littering'*, *'open dumping'*, *'pile of garbage'*) or increase ngram range to `(1, 3)` with sublinear TF-IDF and character n-grams `(3, 5)` to capture morphological cues.

### 4. Experiment 4: Department-Aware Hierarchical Loss Function
- **Diagnostic Basis**: 63% of subcategory regressions were absorbed within the same department. However, 28 records crossed from Waste Management to Sanitation.
- **Proposed Intervention**: Weight misclassification loss asymmetrically: misclassifications between subcategories that share the same department receive a penalty of $1.0$, while cross-department misclassifications receive a penalty of $2.5$.

