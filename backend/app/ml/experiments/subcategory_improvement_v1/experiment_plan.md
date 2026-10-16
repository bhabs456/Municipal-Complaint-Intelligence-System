# MCIS Stage 2 — Experiment Plan: Subcategory Classification Improvement

## 1. Experimental Objectives & Quantitative Targets
- **Primary Accuracy Target**: **80.0%** End-to-End Subcategory Classification Accuracy.
- **Stretch Accuracy Target**: **85.0%** End-to-End Subcategory Classification Accuracy.
- **Baseline Benchmarks (Audited on 1,289 Common Test Set)**:
  - Batch 3 LinearSVC: `71.37%` E2E Accuracy, `0.4898` Macro-F1.
  - Dataset Reduction v2 LinearSVC: `69.74%` E2E Accuracy, `0.4900` Macro-F1.
  - Experiment v3 Logistic Regression: `68.66%` E2E Accuracy, `0.4991` Macro-F1.
  - Batch 3 Oracle Accuracy: `78.90%`.

---

## 2. Mathematical Cascade Error Ceiling Analysis
In a two-stage hierarchical model:
$$\text{Accuracy}_{\text{E2E}} = \text{Accuracy}_{\text{Category}} \times \text{Accuracy}_{\text{Branch | Correct Category}}$$

- Current maximum Category Accuracy (Task 2) is **89.5%**.
- Because incorrect category routing sends complaints to a branch that does NOT contain the true subcategory, **at least 10.5% error is irreversibly locked in at Stage 1**.
- For E2E accuracy to achieve **80.0%**, the branch classifiers would need an average Oracle accuracy of **89.4%** across all 24 classes.
- For E2E accuracy to achieve **85.0%**, the branch classifiers would need an average Oracle accuracy of **95.0%**.

---

## 3. Experimental Hypotheses & Protocol
1. **Hypothesis 1 (Feature Granularity)**: Word-level n-grams up to trigrams `(1,3)` and character n-grams `(3,5)` capture morphological variations in civic vocabulary (*"pothole"*, *"desilting"*), reducing intra-branch confusion.
2. **Hypothesis 2 (Branch Specialization)**: Each municipal domain (e.g. `Sewage & Drainage` vs `Roads & Footpaths`) has distinct vocabulary density. Tuning feature extraction per branch outperforms a one-size-fits-all vectorizer.
3. **Hypothesis 3 (Global vs. Hierarchical)**: A global 24-class classifier eliminates cascading category error, but may suffer from severe long-tail confusion across unrelated classes.

---

## 4. Strict Scientific Guardrails
- **Zero Test Contamination**: All feature selection, vocabulary fitting, and hyperparameter tuning performed strictly on the 1,317 validation samples.
- **Cryptographic Immutability**: Raw CSV SHA-256 (`d951dbb484532421801f6cbd7550edaa6f9ab143da7c4d0e835ba574e4e6d5ac`) preserved.
- **Locked Taxonomy**: Preserved 6 categories, 24 subcategories, 9 departments.
- **Statistical Rigor**: 2,000 paired bootstrap resamples required before asserting improvement.
